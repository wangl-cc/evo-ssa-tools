mod hash;
pub use hash::{DefaultHashObjectCache, HashObjectCache};

mod managed;
pub use managed::{ManagedHashCache, ManagedMemoryCache};

#[cfg(feature = "lru")]
mod lru;
#[cfg(feature = "lru")]
pub use lru::{DefaultLruObjectCache, LruObjectCache};
#[cfg(feature = "lru")]
pub use managed::ManagedLruCache;
