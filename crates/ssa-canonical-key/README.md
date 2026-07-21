# ssa-canonical-key

`ssa-canonical-key` defines a small, fixed-width protocol for encoding scientific-computing inputs as stable cache keys.

Core model crates can implement `CanonicalEncode` for their parameter types without depending on a scheduler or cache backend. Scheduling and Python-binding layers can then reuse the same key identity while choosing their own memory or persistent storage.

## Contract

- Every type declares one fixed encoded width through `CanonicalEncode::SIZE`.
- `CanonicalWriter` prevents writes beyond that width and checks that every nested value writes exactly its declared size.
- `CanonicalBuffer<T>` owns one reusable allocation of exactly `T::SIZE` bytes.
- Built-in signed integers and floating-point values preserve their natural numeric order under unsigned lexicographic byte comparison.
- `usize` and `isize` are encoded as 64-bit values; the crate therefore supports only 64-bit targets.

The writer verifies byte counts, not semantic correctness. Implementations remain responsible for deterministic field order and stable meaning, which should be protected with golden-vector tests and explicit key-format versioning.

## Example

```rust
use ssa_canonical_key::{CanonicalBuffer, CanonicalEncode, CanonicalWriter};

struct Params {
    rate: f64,
    grid: [u16; 2],
}

impl CanonicalEncode for Params {
    const SIZE: usize = f64::SIZE + <[u16; 2]>::SIZE;

    fn encode(&self, writer: &mut CanonicalWriter<'_>) {
        writer.write(&self.rate).write(&self.grid);
    }
}

let mut buffer = CanonicalBuffer::<Params>::new();
let key = buffer.encode(&Params {
    rate: 0.5,
    grid: [16, 32],
});
assert_eq!(key.len(), Params::SIZE);
```
