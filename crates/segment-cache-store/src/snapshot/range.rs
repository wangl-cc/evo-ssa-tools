//! Streaming ordered range cursors.

use std::{cmp::Ordering, sync::Arc};

#[cfg(feature = "value-compression")]
use crate::block::ValuePayloadDecoder;
use crate::{
    block::{DecodedBlock, ParsedRecord},
    error::Result,
    segment::{Segment, SegmentGeometry},
};

/// Streaming cursor over records in key order.
pub struct RangeCursor {
    runs: Vec<SegmentRunCursor>,
    merge_duplicate_indices: Vec<usize>,
}

/// One non-overlapping run of segments traversed with only its current segment loaded.
struct SegmentRunCursor {
    segments: Vec<Arc<Segment>>,
    geometry: SegmentGeometry,
    verify_block_checksums: bool,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    next_segment_index: usize,
    current: Option<SegmentRangeCursor>,
}

impl RangeCursor {
    /// Builds one lazy main run plus one independently mergeable run per patch segment.
    pub(crate) fn from_segment_sets(
        main_segments: Vec<Arc<Segment>>,
        patch_segments: Vec<Arc<Segment>>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Self {
        let mut runs =
            Vec::with_capacity(usize::from(!main_segments.is_empty()) + patch_segments.len());
        if !main_segments.is_empty() {
            runs.push(SegmentRunCursor::new(
                main_segments,
                geometry,
                verify_block_checksums,
                start.clone(),
                end.clone(),
            ));
        }
        for segment in patch_segments {
            runs.push(SegmentRunCursor::new(
                vec![segment],
                geometry,
                verify_block_checksums,
                start.clone(),
                end.clone(),
            ));
        }
        let merge_duplicate_indices = Vec::with_capacity(runs.len());
        Self {
            runs,
            merge_duplicate_indices,
        }
    }

    pub(crate) fn current_record(&mut self) -> Result<Option<ParsedRecord<'_, '_>>> {
        self.ensure_runs_positioned()?;
        let Some(winner_index) = self.refresh_merge_winner_indices()? else {
            return Ok(None);
        };
        self.runs[winner_index].current_record()
    }

    pub(crate) fn advance_record(&mut self) -> Result<()> {
        if self.merge_duplicate_indices.is_empty() {
            self.ensure_runs_positioned()?;
            if self.refresh_merge_winner_indices()?.is_none() {
                return Ok(());
            }
        }
        let mut duplicate_indices = std::mem::take(&mut self.merge_duplicate_indices);
        for index in duplicate_indices.drain(..) {
            self.runs[index].advance()?;
        }
        self.merge_duplicate_indices = duplicate_indices;
        Ok(())
    }

    fn consume_next_record<F>(&mut self, visitor: F) -> Result<bool>
    where
        F: FnOnce(&[u8], &[u8]) -> Result<()>,
    {
        self.ensure_runs_positioned()?;
        let mut duplicate_indices = std::mem::take(&mut self.merge_duplicate_indices);
        let Some(winner_index) = self.next_merged_indices_into(&mut duplicate_indices)? else {
            self.merge_duplicate_indices = duplicate_indices;
            return Ok(false);
        };
        {
            let record = self.runs[winner_index]
                .current_record()?
                .expect("winner run has a current record");
            visitor(record.key, record.value)?;
        }
        for index in duplicate_indices.drain(..) {
            self.runs[index].advance()?;
        }
        self.merge_duplicate_indices = duplicate_indices;
        Ok(true)
    }

    /// Visits all records without allocating owned key/value pairs.
    pub fn visit_all<F>(mut self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]),
    {
        while self.consume_next_record(|key, value| {
            visitor(key, value);
            Ok(())
        })? {
            // Work is done by `consume_next_record`.
        }
        Ok(())
    }

    fn ensure_runs_positioned(&mut self) -> Result<()> {
        for run in &mut self.runs {
            run.ensure_positioned()?;
        }
        Ok(())
    }

    fn refresh_merge_winner_indices(&mut self) -> Result<Option<usize>> {
        let mut duplicate_indices = std::mem::take(&mut self.merge_duplicate_indices);
        let winner_index = self.next_merged_indices_into(&mut duplicate_indices)?;
        self.merge_duplicate_indices = duplicate_indices;
        Ok(winner_index)
    }

    fn next_merged_indices_into(
        &self,
        duplicate_indices: &mut Vec<usize>,
    ) -> Result<Option<usize>> {
        duplicate_indices.clear();
        let mut winner_index = None;
        let mut winner_key = None;
        let mut winner_value = None;
        for (index, run) in self.runs.iter().enumerate() {
            let Some(record) = run.current_record()? else {
                continue;
            };
            let Some(key) = winner_key else {
                winner_index = Some(index);
                winner_key = Some(record.key);
                winner_value = Some(record.value);
                duplicate_indices.push(index);
                continue;
            };
            match record.key.cmp(key) {
                std::cmp::Ordering::Less => {
                    winner_index = Some(index);
                    winner_key = Some(record.key);
                    winner_value = Some(record.value);
                    duplicate_indices.clear();
                    duplicate_indices.push(index);
                }
                std::cmp::Ordering::Equal => {
                    duplicate_indices.push(index);
                    if record.value < winner_value.expect("winner value is set with winner key") {
                        winner_index = Some(index);
                        winner_value = Some(record.value);
                    }
                }
                std::cmp::Ordering::Greater => {}
            }
        }
        Ok(winner_index)
    }
}

impl SegmentRunCursor {
    fn new(
        segments: Vec<Arc<Segment>>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Self {
        Self {
            segments,
            geometry,
            verify_block_checksums,
            start,
            end,
            next_segment_index: 0,
            current: None,
        }
    }

    fn ensure_positioned(&mut self) -> Result<()> {
        loop {
            if self
                .current
                .as_ref()
                .is_some_and(|cursor| !cursor.is_exhausted())
            {
                return Ok(());
            }
            self.current = None;
            let Some(segment) = self.segments.get(self.next_segment_index) else {
                return Ok(());
            };
            self.next_segment_index += 1;
            self.current = Some(SegmentRangeCursor::new(
                Arc::clone(segment),
                self.geometry,
                self.verify_block_checksums,
                self.start.clone(),
                self.end.clone(),
            )?);
        }
    }

    fn current_record(&self) -> Result<Option<ParsedRecord<'_, '_>>> {
        match &self.current {
            Some(cursor) => cursor.current_record(),
            None => Ok(None),
        }
    }

    fn advance(&mut self) -> Result<()> {
        self.current
            .as_mut()
            .expect("winner run is positioned")
            .advance()
    }
}

impl Iterator for RangeCursor {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut item = None;
        match self.consume_next_record(|key, value| {
            item = Some((key.to_vec(), value.to_vec()));
            Ok(())
        }) {
            Ok(true) => Some(Ok(item.expect("visitor stored an item"))),
            Ok(false) => None,
            Err(error) => Some(Err(error)),
        }
    }
}

pub(crate) struct SegmentRangeCursor {
    segment: Arc<Segment>,
    geometry: SegmentGeometry,
    verify_block_checksums: bool,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    block_index: usize,
    position: Option<SegmentCursorPosition>,
    key_scratch: Vec<u8>,
    spare_block_bytes: Vec<u8>,
    #[cfg(feature = "value-compression")]
    payload_decoder: ValuePayloadDecoder,
}

struct SegmentCursorPosition {
    block: DecodedBlock,
    #[cfg(feature = "value-compression")]
    decoded_payload: Option<Vec<u8>>,
    record_index: usize,
}

impl SegmentRangeCursor {
    pub(crate) fn new(
        segment: Arc<Segment>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Result<Self> {
        let block_index = start
            .as_deref()
            .map_or(0, |start| segment.find_block_index(start));
        let mut cursor = Self {
            segment,
            geometry,
            verify_block_checksums,
            start,
            end,
            block_index,
            position: None,
            key_scratch: vec![0; geometry.key_len],
            spare_block_bytes: Vec::new(),
            #[cfg(feature = "value-compression")]
            payload_decoder: ValuePayloadDecoder::new(geometry.value_payload_compression),
        };
        cursor.load_next_valid_record()?;
        Ok(cursor)
    }

    fn current_record(&self) -> Result<Option<ParsedRecord<'_, '_>>> {
        let Some(position) = &self.position else {
            return Ok(None);
        };
        #[cfg(feature = "value-compression")]
        let payload = position
            .block
            .payload_bytes(position.decoded_payload.as_deref())?;
        #[cfg(not(feature = "value-compression"))]
        let payload = position.block.payload_bytes()?;
        let value = position
            .block
            .value_at_index_with_payload(position.record_index, payload)?;
        Ok(Some(ParsedRecord {
            key: &self.key_scratch,
            value,
        }))
    }

    fn advance(&mut self) -> Result<()> {
        let end = self.end.as_deref();
        let has_next_record = match &mut self.position {
            Some(SegmentCursorPosition {
                block,
                record_index,
                ..
            }) => {
                *record_index += 1;
                if *record_index < block.record_count() {
                    let before_end = Self::record_is_before_end(end, block, *record_index);
                    if before_end {
                        block.write_key_at_index(*record_index, &mut self.key_scratch)?;
                    }
                    Some(before_end)
                } else {
                    None
                }
            }
            None => return Ok(()),
        };
        match has_next_record {
            Some(true) => Ok(()),
            Some(false) => {
                self.recycle_current_block();
                Ok(())
            }
            None => self.load_next_valid_record(),
        }
    }

    fn load_next_valid_record(&mut self) -> Result<()> {
        self.recycle_current_block();

        while self.block_index < self.segment.block_count() {
            let block_index = self.block_index;
            if let Some(end) = self.end.as_deref()
                && self.segment.block_min_cmp(block_index, end) != Ordering::Less
            {
                return Ok(());
            }
            if let Some(start) = self.start.as_deref()
                && self.segment.block_max_cmp(block_index, start) == Ordering::Less
            {
                self.block_index += 1;
                continue;
            }
            self.block_index += 1;
            let buffer = std::mem::take(&mut self.spare_block_bytes);
            let block = match self.segment.load_block_reusing(
                block_index,
                self.geometry,
                self.verify_block_checksums,
                buffer,
            ) {
                Ok(block) => block,
                Err(error) if error.is_cache_miss_corruption() => continue,
                Err(error) => return Err(error),
            };

            let record_index = self
                .start
                .as_deref()
                .map_or(0, |start| block.lower_bound_index(start));
            if record_index < block.record_count()
                && Self::record_is_before_end(self.end.as_deref(), &block, record_index)
            {
                if let Err(error) = self.segment.verify_block_payload(
                    block_index,
                    &block,
                    self.verify_block_checksums,
                ) {
                    if error.is_cache_miss_corruption() {
                        self.spare_block_bytes = block.into_bytes();
                        continue;
                    }
                    return Err(error);
                }
                #[cfg(feature = "value-compression")]
                let decoded_payload =
                    match block.decode_payload_if_needed(&mut self.payload_decoder) {
                        Ok(decoded_payload) => decoded_payload,
                        Err(_) => {
                            self.spare_block_bytes = block.into_bytes();
                            continue;
                        }
                    };
                block.write_key_at_index(record_index, &mut self.key_scratch)?;
                self.position = Some(SegmentCursorPosition {
                    block,
                    #[cfg(feature = "value-compression")]
                    decoded_payload,
                    record_index,
                });
                return Ok(());
            }
            self.spare_block_bytes = block.into_bytes();
        }

        Ok(())
    }

    fn record_is_before_end(end: Option<&[u8]>, block: &DecodedBlock, record_index: usize) -> bool {
        if let Some(end) = end {
            return block.compare_key_at_index(record_index, end) == Ordering::Less;
        }
        true
    }

    fn recycle_current_block(&mut self) {
        if let Some(position) = self.position.take() {
            #[cfg(feature = "value-compression")]
            self.payload_decoder
                .reclaim_payload_buffer(position.decoded_payload);
            self.spare_block_bytes = position.block.into_bytes();
        }
    }

    fn is_exhausted(&self) -> bool {
        self.position.is_none()
    }
}
