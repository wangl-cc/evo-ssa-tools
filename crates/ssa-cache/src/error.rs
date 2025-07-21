#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[cfg(feature = "bitcode")]
    #[error("Codec error: {0}")]
    Codec(#[from] bitcode::Error),

    #[cfg(feature = "fjall")]
    #[error("Database error: {0}")]
    Db(#[from] fjall::Error),

    #[error("Try to get cache #{want} but only {total} available")]
    CacheOutofIndex { total: usize, want: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
