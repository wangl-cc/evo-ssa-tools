/// Independent random streams used by one SSA simulation.
///
/// Keeping clock, channel-selection, and model-event randomness separate makes biological draws
/// reproducible when the installed SSA algorithm changes how many scheduler-side draws it consumes.
#[derive(Debug, Clone)]
pub struct SsaRngs<C, S, E> {
    clock: C,
    selection: S,
    event: E,
}

impl<C, S, E> SsaRngs<C, S, E> {
    /// Construct a stream bundle from independently seeded random-number generators.
    pub fn new(clock: C, selection: S, event: E) -> Self {
        Self {
            clock,
            selection,
            event,
        }
    }

    /// Borrow the stream used for waiting times and NRM clock thresholds.
    pub fn clock_mut(&mut self) -> &mut C {
        &mut self.clock
    }

    /// Borrow the stream used for weighted channel selection.
    pub fn selection_mut(&mut self) -> &mut S {
        &mut self.selection
    }

    /// Borrow the stream passed to model reaction effects, such as mutation draws.
    pub fn event_mut(&mut self) -> &mut E {
        &mut self.event
    }

    /// Consume the bundle and return its three streams.
    pub fn into_inner(self) -> (C, S, E) {
        (self.clock, self.selection, self.event)
    }

    pub(crate) fn streams(&mut self) -> (&mut C, &mut S, &mut E) {
        (&mut self.clock, &mut self.selection, &mut self.event)
    }
}
