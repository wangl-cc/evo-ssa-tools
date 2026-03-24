mod hash;
pub use hash::{DefaultHashObjectCache, HashObjectCache};

#[cfg(feature = "lru")]
mod lru;
#[cfg(feature = "lru")]
pub use lru::{DefaultLruObjectCache, LruObjectCache};
