use thiserror::Error;

/// Result type returned by `evo-ssa` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Failures that prevent exact scheduler state from remaining usable.
#[derive(Debug, Error)]
pub enum Error {
    /// A model returned a negative, infinite, or NaN propensity.
    #[error("propensity must be finite and non-negative, got {value}")]
    InvalidPropensity {
        /// Invalid propensity returned by the model.
        value: f64,
    },

    /// A dynamic model requested recomputation of a channel that is not active.
    #[error("cannot recompute a channel that is not active")]
    MissingRecomputeChannel,

    /// A static update named a family outside its branded family bundle.
    #[error("reaction family {family} is outside the configured {family_count} families")]
    InvalidFamily {
        /// Invalid zero-based family index.
        family: usize,
        /// Number of families configured for this model.
        family_count: usize,
    },

    /// A static update named a local channel outside its family's current dense slots.
    #[error(
        "channel {local_channel} is outside reaction family {family}, which currently has {channel_count} channels"
    )]
    InvalidFamilyChannel {
        /// Zero-based family index.
        family: usize,
        /// Invalid family-local channel slot.
        local_channel: usize,
        /// Number of channels currently addressable in the family.
        channel_count: usize,
    },

    /// Cached scheduler structures failed to resolve a selected channel.
    #[error("scheduler selected a missing channel")]
    MissingSelectedChannel,

    /// A previous post-fire update failed, so state and scheduler caches can no longer advance.
    #[error("simulation cannot continue after an event update failed")]
    SimulationPoisoned,
}
