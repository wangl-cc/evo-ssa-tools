use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("propensity must be finite and non-negative, got {value}")]
    InvalidPropensity { value: f64 },

    #[error("scheduler selected a missing channel")]
    MissingSelectedChannel,
}
