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
    /// Per-worker reusable state such as scratch buffers or an assembled operator.
    type State: Default + Send;

    fn compute(
        &self,
        cx: &mut Cx<'_>,
        state: &mut Self::State,
        param: &Self::Param,
        input: &Self::Input,
    ) -> Result<Self::Output>;
}

/// Per-item context. Random streams are derived lazily from (path, label, key), so a
/// deterministic kernel simply never asks for one.
pub struct Cx<'a> {
    seed: &'a [u8; 32],
    key: &'a [u8],
    streams: Vec<(&'static str, Xoshiro256PlusPlus)>,
}

impl<'a> Cx<'a> {
    pub(crate) fn new(seed: &'a [u8; 32], key: &'a [u8]) -> Self {
        Self {
            seed,
            key,
            streams: Vec::new(),
        }
    }

    /// The default random stream.
    pub fn rng(&mut self) -> &mut Xoshiro256PlusPlus {
        self.stream("")
    }

    /// A named random stream. Asking for the same label twice returns the same stream.
    pub fn stream(&mut self, label: &'static str) -> &mut Xoshiro256PlusPlus {
        let index = match self.streams.iter().position(|(name, _)| *name == label) {
            Some(index) => index,
            None => {
                let mut hasher = blake3::Hasher::new_keyed(self.seed);
                hasher
                    .update(label.as_bytes())
                    .update(b"\0")
                    .update(self.key);
                let rng = Xoshiro256PlusPlus::from_seed(hasher.finalize().into());
                self.streams.push((label, rng));
                self.streams.len() - 1
            }
        };
        &mut self.streams[index].1
    }

    /// Canonical key of this item (without namespace).
    pub fn key(&self) -> &[u8] {
        self.key
    }
}
