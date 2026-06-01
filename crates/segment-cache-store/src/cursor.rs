//! Streaming ordered range cursors.

use std::sync::Arc;

use crate::{block::DecodedBlock, error::Result, options::ValueLayout, state::SegmentState};

/// Streaming cursor over records in key order.
pub struct RangeCursor {
    pub(crate) cursors: Vec<SegmentRangeCursor>,
    merge_mode: MergeMode,
}

impl RangeCursor {
    pub(crate) fn new(cursors: Vec<SegmentRangeCursor>, globally_disjoint: bool) -> Self {
        let merge_mode = if globally_disjoint {
            MergeMode::Concatenate {
                active_cursor_index: 0,
            }
        } else {
            MergeMode::KWayMerge
        };
        Self {
            cursors,
            merge_mode,
        }
    }

    /// Visits all records without allocating owned key/value pairs.
    pub fn visit_all<F>(mut self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]),
    {
        while let Some(cursor_index) = self.next_cursor_index() {
            {
                let record = self.cursors[cursor_index]
                    .current_record()?
                    .expect("next_cursor_index only returns cursors with a current record");
                visitor(record.key, record.value);
            }
            self.cursors[cursor_index].advance()?;
            self.advance_active_cursor(cursor_index);
        }
        Ok(())
    }

    fn next_cursor_index(&mut self) -> Option<usize> {
        match &mut self.merge_mode {
            MergeMode::KWayMerge => self
                .cursors
                .iter()
                .enumerate()
                .filter_map(|(index, cursor)| cursor.current_key().map(|key| (index, key)))
                .min_by(|(_, left), (_, right)| left.cmp(right))
                .map(|(index, _)| index),
            MergeMode::Concatenate {
                active_cursor_index,
            } => {
                while *active_cursor_index < self.cursors.len() {
                    if self.cursors[*active_cursor_index].current_key().is_some() {
                        return Some(*active_cursor_index);
                    }
                    *active_cursor_index += 1;
                }
                None
            }
        }
    }

    fn advance_active_cursor(&mut self, cursor_index: usize) {
        if let MergeMode::Concatenate {
            active_cursor_index,
        } = &mut self.merge_mode
            && *active_cursor_index == cursor_index
            && self.cursors[cursor_index].current_key().is_none()
        {
            *active_cursor_index += 1;
        }
    }
}

impl Iterator for RangeCursor {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let cursor_index = self.next_cursor_index()?;
        let record = self.cursors[cursor_index].next_record();
        self.advance_active_cursor(cursor_index);
        Some(record)
    }
}

#[derive(Clone, Copy)]
enum MergeMode {
    KWayMerge,
    Concatenate { active_cursor_index: usize },
}

pub(crate) struct SegmentRangeCursor {
    segment: Arc<SegmentState>,
    key_len: usize,
    value_layout: ValueLayout,
    verify_block_checksums: bool,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    block_index: usize,
    current_block: Option<DecodedBlock>,
    spare_block_bytes: Vec<u8>,
    record_index: usize,
    exhausted: bool,
}

impl SegmentRangeCursor {
    pub(crate) fn new(
        segment: Arc<SegmentState>,
        key_len: usize,
        value_layout: ValueLayout,
        verify_block_checksums: bool,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Result<Self> {
        let block_index = start
            .as_deref()
            .map_or(0, |start| segment.find_block_index(start));
        let mut cursor = Self {
            segment,
            key_len,
            value_layout,
            verify_block_checksums,
            start,
            end,
            block_index,
            current_block: None,
            spare_block_bytes: Vec::new(),
            record_index: 0,
            exhausted: false,
        };
        cursor.load_next_valid_record()?;
        Ok(cursor)
    }

    fn current_key(&self) -> Option<&[u8]> {
        self.current_record()
            .ok()
            .flatten()
            .map(|record| record.key)
    }

    fn next_record(&mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        let record = self
            .current_record()?
            .expect("next_record is only called when current_key exists");
        let key = record.key.to_vec();
        let value = record.value.to_vec();
        self.advance()?;
        Ok((key, value))
    }

    fn current_record(&self) -> Result<Option<crate::block::ParsedRecord<'_>>> {
        if self.exhausted {
            return Ok(None);
        }
        let Some(block) = self.current_block.as_ref() else {
            return Ok(None);
        };
        Ok(Some(block.record_at_index(self.record_index)?))
    }

    fn advance(&mut self) -> Result<()> {
        self.record_index += 1;
        if let Some(block) = self.current_block.as_ref()
            && self.record_index < block.record_count()
        {
            if self.record_is_before_end(block, self.record_index)? {
                return Ok(());
            }
            self.exhausted = true;
            self.current_block = None;
            return Ok(());
        }

        self.load_next_valid_record()
    }

    fn load_next_valid_record(&mut self) -> Result<()> {
        if let Some(block) = self.current_block.take() {
            self.spare_block_bytes = block.into_bytes();
        }
        self.record_index = 0;

        while self.block_index < self.segment.block_index.len() {
            let block_index = self.block_index;
            self.block_index += 1;
            let buffer = std::mem::take(&mut self.spare_block_bytes);
            let block = match self.segment.load_block_reusing(
                block_index,
                self.key_len,
                self.value_layout,
                self.verify_block_checksums,
                buffer,
            ) {
                Ok(block) => block,
                Err(error) if error.is_cache_miss_corruption() => continue,
                Err(error) => return Err(error),
            };

            if let Some(end) = self.end.as_deref()
                && block.first_key.as_slice() >= end
            {
                self.exhausted = true;
                return Ok(());
            }
            if let Some(start) = self.start.as_deref()
                && block.last_key.as_slice() < start
            {
                continue;
            }

            let record_index = self
                .start
                .as_deref()
                .map_or(0, |start| block.lower_bound_index(start));
            if record_index < block.record_count()
                && self.record_is_before_end(&block, record_index)?
            {
                self.current_block = Some(block);
                self.record_index = record_index;
                return Ok(());
            }
        }

        self.exhausted = true;
        Ok(())
    }

    fn record_is_before_end(&self, block: &DecodedBlock, record_index: usize) -> Result<bool> {
        if let Some(end) = self.end.as_deref() {
            return Ok(block.key_at_index(record_index)? < end);
        }
        Ok(true)
    }
}
