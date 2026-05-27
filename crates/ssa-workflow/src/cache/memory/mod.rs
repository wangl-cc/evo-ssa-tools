mod hash;
pub use hash::{DefaultHashObjectCache, HashObjectCache};

mod managed;
pub use managed::{HashMemory, ManagedHashCache, ManagedMemoryCache, MemoryCacheStorage};

#[cfg(feature = "lru")]
mod lru;
#[cfg(feature = "lru")]
pub use lru::{DefaultLruObjectCache, LruObjectCache};
#[cfg(feature = "lru")]
pub use managed::{LruMemory, ManagedLruCache};
