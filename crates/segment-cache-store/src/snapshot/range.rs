//! Streaming ordered range cursors.

use std::{cmp::Ordering, sync::Arc};

use super::CorruptionHandling;
#[cfg(feature = "value-compression")]
use crate::block::ValuePayloadDecoder;
use crate::{
    block::{DecodedBlock, ParsedRecord},
    error::{CorruptionError, Result},
    segment::{Segment, SegmentGeometry},
};

/// Streaming cursor over records in key order.
pub struct RangeCursor {
    runs: Vec<SegmentRunCursor>,
    current_run_index: Option<usize>,
}

/// One non-overlapping run of segments traversed with only its current segment loaded.
struct SegmentRunCursor {
    segments: Vec<Arc<Segment>>,
    geometry: SegmentGeometry,
    verify_block_checksums: bool,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    corruption_handling: CorruptionHandling,
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
        Self::new(
            main_segments,
            patch_segments,
            geometry,
            verify_block_checksums,
            start,
            end,
            CorruptionHandling::AsCacheMiss,
        )
    }

    pub(crate) fn strict_from_segment_sets(
        main_segments: Vec<Arc<Segment>>,
        patch_segments: Vec<Arc<Segment>>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Self {
        Self::new(
            main_segments,
            patch_segments,
            geometry,
            verify_block_checksums,
            start,
            end,
            CorruptionHandling::Strict,
        )
    }

    fn new(
        main_segments: Vec<Arc<Segment>>,
        patch_segments: Vec<Arc<Segment>>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
        corruption_handling: CorruptionHandling,
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
                corruption_handling,
            ));
        }
        for segment in patch_segments {
            runs.push(SegmentRunCursor::new(
                vec![segment],
                geometry,
                verify_block_checksums,
                start.clone(),
                end.clone(),
                corruption_handling,
            ));
        }
        Self {
            runs,
            current_run_index: None,
        }
    }

    pub(crate) fn current_record(&mut self) -> Result<Option<ParsedRecord<'_, '_>>> {
        if self.current_run_index.is_none() {
            self.ensure_runs_positioned()?;
            self.current_run_index = self.next_run_index()?;
        }
        let Some(run_index) = self.current_run_index else {
            return Ok(None);
        };
        self.runs[run_index].current_record()
    }

    pub(crate) fn advance_record(&mut self) -> Result<()> {
        if self.current_run_index.is_none() {
            self.ensure_runs_positioned()?;
            self.current_run_index = self.next_run_index()?;
        }
        let Some(run_index) = self.current_run_index.take() else {
            return Ok(());
        };
        self.runs[run_index].advance()?;
        Ok(())
    }

    fn consume_next_record<F>(&mut self, visitor: F) -> Result<bool>
    where
        F: FnOnce(&[u8], &[u8]) -> Result<()>,
    {
        self.ensure_runs_positioned()?;
        let Some(run_index) = self.next_run_index()? else {
            return Ok(false);
        };
        {
            let record = self.runs[run_index]
                .current_record()?
                .expect("selected run has a current record");
            visitor(record.key, record.value)?;
        }
        self.runs[run_index].advance()?;
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

    fn next_run_index(&self) -> Result<Option<usize>> {
        let mut selected_index = None;
        let mut selected_key = None;
        for (index, run) in self.runs.iter().enumerate() {
            let Some(record) = run.current_record()? else {
                continue;
            };
            let Some(key) = selected_key else {
                selected_index = Some(index);
                selected_key = Some(record.key);
                continue;
            };
            match record.key.cmp(key) {
                Ordering::Less => {
                    selected_index = Some(index);
                    selected_key = Some(record.key);
                }
                Ordering::Equal => return Err(CorruptionError::DuplicateVisibleKey.into()),
                Ordering::Greater => {}
            }
        }
        Ok(selected_index)
    }
}

impl SegmentRunCursor {
    fn new(
        segments: Vec<Arc<Segment>>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
        corruption_handling: CorruptionHandling,
    ) -> Self {
        Self {
            segments,
            geometry,
            verify_block_checksums,
            start,
            end,
            corruption_handling,
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
                self.corruption_handling,
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
            .expect("selected run is positioned")
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
    corruption_handling: CorruptionHandling,
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
    record_end: usize,
}

impl SegmentRangeCursor {
    pub(crate) fn new(
        segment: Arc<Segment>,
        geometry: SegmentGeometry,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
        corruption_handling: CorruptionHandling,
    ) -> Result<Self> {
        let mut exhausted = false;
        let start = match start {
            Some(start) if start.as_slice() > segment.max_key() => {
                exhausted = true;
                None
            }
            Some(start) if start.as_slice() <= segment.min_key() => None,
            start => start,
        };
        let end = match end {
            Some(end) if end.as_slice() <= segment.min_key() => {
                exhausted = true;
                None
            }
            Some(end) if end.as_slice() > segment.max_key() => None,
            end => end,
        };
        let block_index = if exhausted {
            segment.block_count()
        } else {
            start.as_deref().map_or(0, |start| {
                segment.find_block_index(segment.relative_key_in_range(start))
            })
        };
        let mut cursor = Self {
            segment,
            geometry,
            verify_block_checksums,
            start,
            end,
            corruption_handling,
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
        let segment_prefix = self.segment.segment_prefix();
        let has_next_record = match &mut self.position {
            Some(SegmentCursorPosition {
                block,
                record_index,
                record_end,
                ..
            }) => {
                *record_index += 1;
                if *record_index < *record_end {
                    block.write_key_at_index(
                        segment_prefix,
                        *record_index,
                        &mut self.key_scratch,
                    )?;
                    Some(true)
                } else if *record_end < block.record_count() {
                    Some(false)
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
                && self
                    .segment
                    .block_min_cmp(block_index, self.segment.relative_key_in_range(end))
                    != Ordering::Less
            {
                return Ok(());
            }
            if let Some(start) = self.start.as_deref()
                && self
                    .segment
                    .block_max_cmp(block_index, self.segment.relative_key_in_range(start))
                    == Ordering::Less
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
                Err(error) if self.corruption_handling.degrades(&error) => continue,
                Err(error) => return Err(error),
            };

            let record_index = self.start.as_deref().map_or(0, |start| {
                let key = self.segment.relative_key_in_range(start);
                if self.segment.block_min_cmp(block_index, key) != Ordering::Less {
                    0
                } else {
                    block.lower_bound_index(self.segment.block_relative_key(block_index, key))
                }
            });
            let record_end = self.end.as_deref().map_or(block.record_count(), |end| {
                let key = self.segment.relative_key_in_range(end);
                if self.segment.block_max_cmp(block_index, key) == Ordering::Less {
                    block.record_count()
                } else {
                    block.lower_bound_index(self.segment.block_relative_key(block_index, key))
                }
            });
            if record_index < record_end {
                if let Err(error) = self.segment.verify_block_payload(
                    block_index,
                    &block,
                    self.verify_block_checksums,
                ) {
                    if self.corruption_handling.degrades(&error) {
                        self.spare_block_bytes = block.into_bytes();
                        continue;
                    }
                    return Err(error);
                }
                #[cfg(feature = "value-compression")]
                let decoded_payload =
                    match block.decode_payload_if_needed(&mut self.payload_decoder) {
                        Ok(decoded_payload) => decoded_payload,
                        Err(_) if self.corruption_handling.degrades_corruption() => {
                            self.spare_block_bytes = block.into_bytes();
                            continue;
                        }
                        Err(error) => return Err(error.into()),
                    };
                block.write_key_at_index(
                    self.segment.segment_prefix(),
                    record_index,
                    &mut self.key_scratch,
                )?;
                self.position = Some(SegmentCursorPosition {
                    block,
                    #[cfg(feature = "value-compression")]
                    decoded_payload,
                    record_index,
                    record_end,
                });
                return Ok(());
            }
            self.spare_block_bytes = block.into_bytes();
        }

        Ok(())
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
