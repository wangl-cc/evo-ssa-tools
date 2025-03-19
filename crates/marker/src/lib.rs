mod util;

/// A trait used to represent a marker that can be used to trace the lineage of cells.
pub trait Marker: Sized {
    /// Global state of the simulation used to generate new markers
    type State;

    /// Called when a cell divides
    ///
    /// There are two daughter cells whose markers are derived from the mother cell.
    /// One daughter cell should replace the mother cell, the another should be returned.
    fn divide(&mut self, state: &mut Self::State) -> Self {
        let daughter1 = self.gen_marker(state);
        let daughter2 = self.gen_marker(state);
        *self = daughter1;
        daughter2
    }

    /// Generate a new marker based on the current marker and the global state
    ///
    /// Do not use this method directly. Use the `divide` method instead during cell division.
    fn gen_marker(&self, state: &mut Self::State) -> Self;
}

/// A placeholder marker that when you don't want to use markers.
///
/// Rust compiler should optimize away the marker and its associated code.
/// So this is zero overhead, and should work like nothing happened.
type NoMarker = ();

impl Marker for NoMarker {
    type State = ();

    // Override the default implementation to make sure this method does nothing
    fn divide(&mut self, _: &mut Self::State) -> Self {}

    fn gen_marker(&self, _: &mut Self::State) -> Self {}
}

// Use point mutations as marker
pub mod mutations;
