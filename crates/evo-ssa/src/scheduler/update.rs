/// A channel update addressed by reaction-family id and local channel id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyChannelUpdate {
    pub family: usize,
    pub local_channel: usize,
}

/// Callback sink used by statically dispatched reaction families to report affected channels.
///
/// `recompute` addresses a channel in the currently firing family. `recompute_family` is the
/// preferred form for family-separated schedulers. `recompute_global` exists for flattened static
/// direct schedulers and compatibility with older benchmark code.
pub trait StaticUpdateSink {
    fn recompute(&mut self, local_channel: usize);

    fn recompute_global(&mut self, channel: usize);

    fn recompute_family(&mut self, family: usize, local_channel: usize);
}

/// Heap-allocated update buffer used by generic family-separated schedulers.
pub struct FamilyChannelUpdates<'a> {
    family_offsets: &'a [usize],
    updates: &'a mut Vec<FamilyChannelUpdate>,
}

pub(crate) trait FamilyChannelUpdateSink {
    fn push_update(&mut self, update: FamilyChannelUpdate);
}

impl FamilyChannelUpdates<'_> {
    pub fn recompute(&mut self, family: usize, local_channel: usize) {
        self.push_update(FamilyChannelUpdate {
            family,
            local_channel,
        });
    }

    pub fn recompute_global(&mut self, channel: usize) {
        if let Some((family, local_channel)) = global_to_family(self.family_offsets, channel) {
            self.recompute(family, local_channel);
        }
    }

    pub(crate) fn new<'a>(
        family_offsets: &'a [usize],
        updates: &'a mut Vec<FamilyChannelUpdate>,
    ) -> FamilyChannelUpdates<'a> {
        FamilyChannelUpdates {
            family_offsets,
            updates,
        }
    }

    pub(crate) fn writer(&mut self, family: usize, offset: usize) -> StaticChannelUpdates<'_> {
        StaticChannelUpdates::family(family, offset, self.family_offsets, self)
    }

    pub(crate) fn offset_for(&self, family: usize) -> Option<usize> {
        self.family_offsets.get(family).copied()
    }
}

impl FamilyChannelUpdateSink for FamilyChannelUpdates<'_> {
    fn push_update(&mut self, update: FamilyChannelUpdate) {
        self.updates.push(update);
    }
}

/// Fixed-capacity update buffer for hot schedulers with known small dependency fanout.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct FixedFamilyChannelUpdates<const N: usize> {
    updates: [FamilyChannelUpdate; N],
    len: usize,
}

impl<const N: usize> Default for FixedFamilyChannelUpdates<N> {
    fn default() -> Self {
        Self {
            updates: [FamilyChannelUpdate {
                family: 0,
                local_channel: 0,
            }; N],
            len: 0,
        }
    }
}

impl<const N: usize> FixedFamilyChannelUpdates<N> {
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    pub(crate) fn writer<'a>(
        &'a mut self,
        family: usize,
        _offset: usize,
        family_offsets: &'a [usize],
    ) -> FamilyLocalUpdateWriter<'a, Self> {
        FamilyLocalUpdateWriter {
            family,
            family_offsets,
            sink: self,
        }
    }

    pub(crate) fn pop(&mut self) -> Option<FamilyChannelUpdate> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.updates[self.len])
    }
}

impl<const N: usize> FamilyChannelUpdateSink for FixedFamilyChannelUpdates<N> {
    fn push_update(&mut self, update: FamilyChannelUpdate) {
        assert!(
            self.len < N,
            "fixed family-channel update buffer capacity exceeded"
        );
        self.updates[self.len] = update;
        self.len += 1;
    }
}

pub(crate) struct FamilyLocalUpdateWriter<'a, S> {
    family: usize,
    family_offsets: &'a [usize],
    sink: &'a mut S,
}

impl<S> StaticUpdateSink for FamilyLocalUpdateWriter<'_, S>
where
    S: FamilyChannelUpdateSink,
{
    fn recompute(&mut self, local_channel: usize) {
        self.sink.push_update(FamilyChannelUpdate {
            family: self.family,
            local_channel,
        });
    }

    fn recompute_global(&mut self, channel: usize) {
        if let Some((family, local_channel)) = global_to_family(self.family_offsets, channel) {
            self.sink.push_update(FamilyChannelUpdate {
                family,
                local_channel,
            });
        }
    }

    fn recompute_family(&mut self, family: usize, local_channel: usize) {
        self.sink.push_update(FamilyChannelUpdate {
            family,
            local_channel,
        });
    }
}

/// Update writer passed to static reaction families by flattened static schedulers.
pub struct StaticChannelUpdates<'a> {
    family: usize,
    offset: usize,
    family_offsets: &'a [usize],
    sink: StaticChannelUpdateSink<'a>,
}

enum StaticChannelUpdateSink<'a> {
    Global(&'a mut Vec<usize>),
    Family(&'a mut dyn FamilyChannelUpdateSink),
}

impl StaticUpdateSink for StaticChannelUpdates<'_> {
    fn recompute(&mut self, local_channel: usize) {
        match &mut self.sink {
            StaticChannelUpdateSink::Global(updates) => {
                updates.push(self.offset + local_channel);
            }
            StaticChannelUpdateSink::Family(updates) => {
                updates.push_update(FamilyChannelUpdate {
                    family: self.family,
                    local_channel,
                });
            }
        }
    }

    fn recompute_global(&mut self, channel: usize) {
        match &mut self.sink {
            StaticChannelUpdateSink::Global(updates) => updates.push(channel),
            StaticChannelUpdateSink::Family(updates) => {
                if let Some((family, local_channel)) =
                    global_to_family(self.family_offsets, channel)
                {
                    updates.push_update(FamilyChannelUpdate {
                        family,
                        local_channel,
                    });
                }
            }
        }
    }

    fn recompute_family(&mut self, family: usize, local_channel: usize) {
        match &mut self.sink {
            StaticChannelUpdateSink::Global(updates) => {
                if let Some(offset) = self.family_offsets.get(family) {
                    updates.push(offset + local_channel);
                }
            }
            StaticChannelUpdateSink::Family(updates) => {
                updates.push_update(FamilyChannelUpdate {
                    family,
                    local_channel,
                });
            }
        }
    }
}

impl StaticChannelUpdates<'_> {
    pub(crate) fn global<'a>(
        family: usize,
        offset: usize,
        family_offsets: &'a [usize],
        updates: &'a mut Vec<usize>,
    ) -> StaticChannelUpdates<'a> {
        StaticChannelUpdates {
            family,
            offset,
            family_offsets,
            sink: StaticChannelUpdateSink::Global(updates),
        }
    }

    fn family<'a>(
        family: usize,
        offset: usize,
        family_offsets: &'a [usize],
        updates: &'a mut dyn FamilyChannelUpdateSink,
    ) -> StaticChannelUpdates<'a> {
        StaticChannelUpdates {
            family,
            offset,
            family_offsets,
            sink: StaticChannelUpdateSink::Family(updates),
        }
    }
}

pub(crate) fn global_to_family(offsets: &[usize], global_channel: usize) -> Option<(usize, usize)> {
    let family = offsets.partition_point(|&offset| offset <= global_channel);
    let family = family.checked_sub(1)?;
    Some((family, global_channel - offsets[family]))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn static_channel_updates_global_writer_records_global_ids() {
        let family_offsets = [0, 3, 8];
        let mut updates = Vec::new();
        let mut writer = StaticChannelUpdates::global(1, 3, &family_offsets, &mut updates);

        writer.recompute(2);
        writer.recompute_family(2, 4);
        writer.recompute_global(7);

        assert_eq!(updates, vec![5, 12, 7]);
    }

    #[test]
    fn fixed_family_updates_writer_records_family_local_ids() {
        let family_offsets = [0, 3, 8];
        let mut updates = FixedFamilyChannelUpdates::<4>::default();
        {
            let mut writer = updates.writer(1, 3, &family_offsets);
            writer.recompute(2);
            writer.recompute_family(2, 4);
            writer.recompute_global(7);
        }

        assert_eq!(
            updates.pop(),
            Some(FamilyChannelUpdate {
                family: 1,
                local_channel: 4,
            })
        );
        assert_eq!(
            updates.pop(),
            Some(FamilyChannelUpdate {
                family: 2,
                local_channel: 4,
            })
        );
        assert_eq!(
            updates.pop(),
            Some(FamilyChannelUpdate {
                family: 1,
                local_channel: 2,
            })
        );
        assert_eq!(updates.pop(), None);
    }
}
