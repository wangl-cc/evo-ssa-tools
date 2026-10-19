/// Errors produced by the prototype.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The upstream passed to a request does not come from the stage this stage was bound after.
    #[error("upstream path mismatch: stage expects `{expected}`, got `{actual}`")]
    WrongUpstream { expected: String, actual: String },

    /// A per-item parameter list does not match the upstream length.
    #[error("expected {expected} parameters, got {actual}")]
    ParamCount { expected: usize, actual: usize },

    /// A stored value could not be decoded. This is a hard error, not a miss.
    #[error("failed to decode a value of `{path}`: {message}")]
    Decode { path: String, message: String },

    /// A kernel failed.
    #[error("kernel `{path}` failed: {message}")]
    Kernel { path: String, message: String },
}

/// Result type of the prototype.
pub type Result<T, E = Error> = std::result::Result<T, E>;
