//! Streaming ordered range cursors.

use std::sync::Arc;

use crate::{
    engine::runtime::SegmentState,
    error::Result,
    format::{
        ValueLayout,
        block::{DecodedBlock, ParsedRecord},
    },
};

/// Streaming cursor over records in key order.
pub struct RangeCursor {
    pub(crate) cursors: Vec<SegmentRangeCursor>,
    mode: RangeCursorMode,
}

enum RangeCursorMode {
    Concatenate { active_cursor_index: usize },
    Merge,
}

impl RangeCursor {
    pub(crate) fn new(cursors: Vec<SegmentRangeCursor>) -> Self {
        Self {
            cursors,
            mode: RangeCursorMode::Concatenate {
                active_cursor_index: 0,
            },
        }
    }

    pub(crate) fn merge(cursors: Vec<SegmentRangeCursor>) -> Self {
        Self {
            cursors,
            mode: RangeCursorMode::Merge,
        }
    }

    /// Visits all records without allocating owned key/value pairs.
    pub fn visit_all<F>(mut self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]),
    {
        match self.mode {
            RangeCursorMode::Concatenate { .. } => {
                while let Some(cursor_index) = self.next_concat_cursor_index()? {
                    {
                        let Some(record) = self.cursors[cursor_index].current_record()? else {
                            self.advance_empty_concat_cursor(cursor_index);
                            continue;
                        };
                        visitor(record.key, record.value);
                    }
                    self.cursors[cursor_index].advance()?;
                    self.advance_concat_cursor(cursor_index)?;
                }
            }
            RangeCursorMode::Merge => {
                let mut duplicate_indices = Vec::with_capacity(self.cursors.len());
                while let Some(winner_index) =
                    self.next_merged_indices_into(&mut duplicate_indices)?
                {
                    {
                        let record = self.cursors[winner_index]
                            .current_record()?
                            .expect("winner cursor has a current record");
                        visitor(record.key, record.value);
                    }
                    for index in duplicate_indices.drain(..) {
                        self.cursors[index].advance()?;
                    }
                }
            }
        }
        Ok(())
    }

    fn next_concat_cursor_index(&mut self) -> Result<Option<usize>> {
        let RangeCursorMode::Concatenate {
            active_cursor_index,
        } = &mut self.mode
        else {
            return Ok(None);
        };
        while *active_cursor_index < self.cursors.len() {
            if self.cursors[*active_cursor_index]
                .current_record()?
                .is_some()
            {
                return Ok(Some(*active_cursor_index));
            }
            *active_cursor_index += 1;
        }
        Ok(None)
    }

    fn advance_empty_concat_cursor(&mut self, cursor_index: usize) {
        if let RangeCursorMode::Concatenate {
            active_cursor_index,
        } = &mut self.mode
            && *active_cursor_index == cursor_index
        {
            *active_cursor_index += 1;
        }
    }

    fn advance_concat_cursor(&mut self, cursor_index: usize) -> Result<()> {
        if let RangeCursorMode::Concatenate {
            active_cursor_index,
        } = &mut self.mode
            && *active_cursor_index == cursor_index
            && self.cursors[cursor_index].current_record()?.is_none()
        {
            *active_cursor_index += 1;
        }
        Ok(())
    }

    fn next_merged_record(&mut self) -> Result<Option<OwnedRecord>> {
        let mut duplicate_indices = Vec::with_capacity(self.cursors.len());
        let Some(winner_index) = self.next_merged_indices_into(&mut duplicate_indices)? else {
            return Ok(None);
        };
        let record = {
            let record = self.cursors[winner_index]
                .current_record()?
                .expect("winner cursor has a current record");
            OwnedRecord {
                key: record.key.to_vec(),
                value: record.value.to_vec(),
            }
        };
        for index in duplicate_indices {
            self.cursors[index].advance()?;
        }
        Ok(Some(record))
    }

    fn next_merged_indices_into(
        &self,
        duplicate_indices: &mut Vec<usize>,
    ) -> Result<Option<usize>> {
        duplicate_indices.clear();
        let mut winner_index = None;
        for index in 0..self.cursors.len() {
            let Some(record) = self.cursors[index].current_record()? else {
                continue;
            };
            let Some(winner) = winner_index else {
                winner_index = Some(index);
                duplicate_indices.push(index);
                continue;
            };
            let winner_record = self.cursors[winner]
                .current_record()?
                .expect("winner cursor has a current record");
            match record.key.cmp(winner_record.key) {
                std::cmp::Ordering::Less => {
                    winner_index = Some(index);
                    duplicate_indices.clear();
                    duplicate_indices.push(index);
                }
                std::cmp::Ordering::Equal => {
                    duplicate_indices.push(index);
                    if record.value < winner_record.value {
                        winner_index = Some(index);
                    }
                }
                std::cmp::Ordering::Greater => {}
            }
        }
        Ok(winner_index)
    }
}

struct OwnedRecord {
    key: Vec<u8>,
    value: Vec<u8>,
}

impl Iterator for RangeCursor {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.mode {
                RangeCursorMode::Concatenate { .. } => {
                    let cursor_index = match self.next_concat_cursor_index() {
                        Ok(Some(cursor_index)) => cursor_index,
                        Ok(None) => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    let record = match self.cursors[cursor_index].current_record() {
                        Ok(Some(record)) => record,
                        Ok(None) => {
                            self.advance_empty_concat_cursor(cursor_index);
                            continue;
                        }
                        Err(error) => return Some(Err(error)),
                    };
                    let item = Ok((record.key.to_vec(), record.value.to_vec()));
                    if let Err(error) = self.cursors[cursor_index].advance() {
                        return Some(Err(error));
                    }
                    if let Err(error) = self.advance_concat_cursor(cursor_index) {
                        return Some(Err(error));
                    }
                    return Some(item);
                }
                RangeCursorMode::Merge => {
                    return match self.next_merged_record() {
                        Ok(Some(record)) => Some(Ok((record.key, record.value))),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    };
                }
            }
        }
    }
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

    fn current_record(&self) -> Result<Option<ParsedRecord<'_>>> {
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
                && block.first_key() >= end
            {
                self.exhausted = true;
                return Ok(());
            }
            if let Some(start) = self.start.as_deref()
                && block.last_key() < start
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
