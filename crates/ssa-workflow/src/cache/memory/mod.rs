mod hash;
pub use hash::{DefaultHashObjectCache, HashObjectCache, ManagedHashCache};

mod managed;
pub use managed::ManagedMemoryCache;

#[cfg(feature = "lru")]
mod lru;
#[cfg(feature = "lru")]
pub use lru::{DefaultLruObjectCache, LruObjectCache, ManagedLruCache};
