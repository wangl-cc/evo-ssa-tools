#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[cfg(feature = "bitcode")]
    #[error("Codec error")]
    Codec(#[from] bitcode::Error),

    #[cfg(feature = "fjall")]
    #[error("Database error")]
    Db(#[from] fjall::Error),

    #[error("Try to get cache #{want} but only {total} available")]
    CacheOutofIndex { total: usize, want: usize },

    #[error("Execution interrupted")]
    Interrupted,

    #[error("Compute error")]
    Compute(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, Error>;
