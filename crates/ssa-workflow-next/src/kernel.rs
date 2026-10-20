use canonical_input_encoding::CanonicalEncode;
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use crate::{Result, Value};

/// One computation. All types come from associated types, so callers never annotate them.
///
/// The output for an item may depend only on the item's key (which contains its parameter and
/// its whole upstream key chain), its parameter and its upstream value. It must not depend on
/// the worker state's history, on which other items are in the block, or on thread counts.
pub trait Kernel: Sync {
    /// Stable, versioned computation id.
    const ID: &'static str;
    /// Encoded into the key in front of the upstream key.
    type Param: CanonicalEncode + Sync;
    /// Upstream output, borrowed. `()` for root stages.
    type Input: Sync;
    /// Output stored in the cache.
    type Output: Value;
    /// Per-worker reusable state such as scratch buffers or an assembled operator. Any
    /// `Default` type works as is; implement [`State`] to build it from the kernel instead.
    type State: State<Self>;

    fn compute(
        &self,
        streams: Streams<'_>,
        state: &mut Self::State,
        param: &Self::Param,
        input: &Self::Input,
    ) -> Result<Self::Output>;
}

/// How a worker state is built. Each worker builds (or reuses) one when it starts.
///
/// Every `Default + Send` type implements this by calling `Default::default`. A state that needs
/// the kernel to build itself (for example to size a thread pool from a kernel field) does not
/// implement `Default` and implements `State<MyKernel>` instead.
///
/// A state is only a cache or scratch space: results must be the same as with a freshly built
/// state, whatever the worker computed before.
pub trait State<K: ?Sized>: Send + Sized {
    fn new(kernel: &K) -> Self;
}

impl<K: ?Sized, T: Default + Send> State<K> for T {
    fn new(_: &K) -> Self {
        T::default()
    }
}

/// Random streams of one item. Each stream is derived from (stage path, label, item key) only
/// when asked for, so a deterministic kernel that ignores this costs nothing: it is two
/// references, with no allocation and no hashing.
#[derive(Clone, Copy)]
pub struct Streams<'a> {
    seed: &'a [u8; 32],
    key: &'a [u8],
}

impl<'a> Streams<'a> {
    pub(crate) fn new(seed: &'a [u8; 32], key: &'a [u8]) -> Self {
        Self { seed, key }
    }

    /// The default random stream.
    pub fn rng(self) -> Xoshiro256PlusPlus {
        self.stream("")
    }

    /// A named random stream, independent of every other label. Each call starts the stream
    /// from its beginning, so take it once and keep it.
    pub fn stream(self, label: &str) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::from_seed(self.seed(label))
    }

    /// The seed behind `stream(label)`, for kernels that bring their own generator. The
    /// framework only derives seeds; the kernel owns and advances its generators.
    pub fn seed(self, label: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_keyed(self.seed);
        hasher
            .update(label.as_bytes())
            .update(b"\0")
            .update(self.key);
        hasher.finalize().into()
    }
}
