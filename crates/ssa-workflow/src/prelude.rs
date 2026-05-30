//! Common imports for end-user workflow definitions.

#[cfg(feature = "derive")]
pub use ssa_workflow_derive::CanonicalEncode;

#[cfg(feature = "bitcode06")]
pub use crate::cache::codec::Bitcode06;
#[cfg(feature = "postcard")]
pub use crate::cache::codec::Postcard;
#[cfg(feature = "compress")]
pub use crate::cache::codec::compress::CompressCodecExt;
#[cfg(feature = "lz4")]
pub use crate::cache::codec::compress::Lz4;
#[cfg(feature = "zstd")]
pub use crate::cache::codec::compress::Zstd;
#[cfg(feature = "lru")]
pub use crate::cache::memory::ManagedLruCache;
#[cfg(feature = "fjall3")]
pub use crate::cache::storage::Fjall3StorageProvider;
pub use crate::{
    Compute, InterruptSignal,
    cache::{CanonicalEncode, StorageProviderExt, memory::ManagedHashCache},
    compute::{
        DependentInput, DependentStochasticInput, DeterministicTask, RandomVariable, RngBundle,
        StochasticInput, StochasticTask, StochasticTransform, Transform, TransformExt,
    },
};
