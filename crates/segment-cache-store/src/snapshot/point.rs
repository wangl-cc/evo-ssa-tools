//! Independent point lookup over one immutable segment snapshot.

use std::sync::Arc;

use super::LookupReadOptions;
#[cfg(feature = "value-compression")]
use crate::block::ValuePayloadDecoder;
use crate::{error::Result, segment::Segment};

pub(crate) struct SegmentSetReader<'a> {
    main_segments: &'a [Arc<Segment>],
    patch_segments: &'a [Arc<Segment>],
    options: LookupReadOptions,
}

impl<'a> SegmentSetReader<'a> {
    pub(crate) fn new(
        main_segments: &'a [Arc<Segment>],
        patch_segments: &'a [Arc<Segment>],
        options: LookupReadOptions,
    ) -> Self {
        Self {
            main_segments,
            patch_segments,
            options,
        }
    }

    pub(crate) fn fetch_one(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut winner = self.fetch_one_from_main(key)?;
        for segment in self.patch_segments {
            if segment.min_key() <= key
                && key <= segment.max_key()
                && let Some(value) = self.fetch_one_from_segment(segment, key)?
                && winner
                    .as_ref()
                    .is_none_or(|winner| value.as_slice() < winner.as_slice())
            {
                winner = Some(value);
            }
        }
        Ok(winner)
    }

    fn fetch_one_from_main(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let segment_index = match self
            .main_segments
            .partition_point(|segment| segment.min_key() <= key)
        {
            0 => return Ok(None),
            idx => idx - 1,
        };
        let segment = &self.main_segments[segment_index];
        if key > segment.max_key() {
            return Ok(None);
        }
        self.fetch_one_from_segment(segment, key)
    }

    fn fetch_one_from_segment(&self, segment: &Segment, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let segment_key = segment.relative_key_in_range(key);
        let block_index = segment.find_block_index(segment_key);
        if !segment.block_contains(block_index, segment_key) {
            return Ok(None);
        }
        let block_key = segment.block_relative_key(block_index, segment_key);
        #[cfg(feature = "value-compression")]
        let mut payload_decoder =
            ValuePayloadDecoder::new(self.options.geometry.value_payload_compression);
        let block = match segment.load_block(
            block_index,
            self.options.geometry,
            self.options.verify_block_checksums,
        ) {
            Ok(block) => block,
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        };
        let record_index = block.lower_bound_index(block_key);
        if !block.key_matches_at_index(record_index, block_key) {
            return Ok(None);
        }
        match segment.verify_block_payload(block_index, &block, self.options.verify_block_checksums)
        {
            Ok(()) => {}
            Err(error) if error.is_cache_miss_corruption() => return Ok(None),
            Err(error) => return Err(error),
        }
        #[cfg(feature = "value-compression")]
        let value = {
            let decoded_payload = match block.decode_payload_if_needed(&mut payload_decoder) {
                Ok(decoded_payload) => decoded_payload,
                Err(_) => return Ok(None),
            };
            block
                .value_at_index(record_index, decoded_payload.as_deref())?
                .to_vec()
        };
        #[cfg(not(feature = "value-compression"))]
        let value = block
            .value_at_index_with_payload(record_index, block.payload_bytes()?)?
            .to_vec();
        Ok(Some(value))
    }
}
