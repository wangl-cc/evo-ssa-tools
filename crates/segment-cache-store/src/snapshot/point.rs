//! Independent point lookup over one immutable segment snapshot.

use std::sync::Arc;

use crate::{
    block::ValuePayloadDecoder, error::Result, segment::state::SegmentState,
    snapshot::LookupReadOptions,
};

pub(crate) struct SegmentSetReader<'a> {
    main_segments: &'a [Arc<SegmentState>],
    patch_segments: &'a [Arc<SegmentState>],
    options: LookupReadOptions,
}

impl<'a> SegmentSetReader<'a> {
    pub(crate) fn new(
        main_segments: &'a [Arc<SegmentState>],
        patch_segments: &'a [Arc<SegmentState>],
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
            if segment.min_key.as_slice() <= key
                && key <= segment.max_key.as_slice()
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
            .partition_point(|segment| segment.min_key.as_slice() <= key)
        {
            0 => return Ok(None),
            idx => idx - 1,
        };
        let segment = &self.main_segments[segment_index];
        if key > segment.max_key.as_slice() {
            return Ok(None);
        }
        self.fetch_one_from_segment(segment, key)
    }

    fn fetch_one_from_segment(
        &self,
        segment: &SegmentState,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>> {
        let block_index = segment.find_block_index(key);
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
        let record_index = block.lower_bound_index(key);
        if !block.key_matches_at_index(record_index, key) {
            return Ok(None);
        }
        let decoded_payload = match block.decode_payload_if_needed(&mut payload_decoder) {
            Ok(decoded_payload) => decoded_payload,
            Err(_) => return Ok(None),
        };
        Ok(block
            .value_at_index(record_index, decoded_payload.as_deref())
            .map(|value| Some(value.to_vec()))?)
    }
}
