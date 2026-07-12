//! Stochastic input identity and lazy repeated input sources.

mod parallel;
mod serial;

pub use parallel::RepeatedStochasticIntoParIter;
pub use serial::RepeatedStochasticIntoIter;

use crate::cache::CanonicalEncode;

/// Input for one stochastic execution.
///
/// This wraps a deterministic parameter value (`param`) with a `repetition_index`.
///
/// The pair serves two roles:
///
/// - It is the canonical input used for cache-key construction.
/// - It selects one reproducible stochastic repetition.
///
/// # Encoding
///
/// Canonical encoding is `param` bytes followed by big-endian `repetition_index` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticInput<P> {
    /// Deterministic model/config input.
    pub param: P,
    /// Repetition index used for random stream derivation.
    pub repetition_index: u64,
}

impl<P> StochasticInput<P> {
    /// Create a stochastic input from `(param, repetition_index)`.
    pub const fn new(param: P, repetition_index: u64) -> Self {
        Self {
            param,
            repetition_index,
        }
    }
}

impl<P> From<(P, u64)> for StochasticInput<P> {
    fn from((param, repetition_index): (P, u64)) -> Self {
        Self {
            param,
            repetition_index,
        }
    }
}

impl<P: CanonicalEncode> CanonicalEncode for StochasticInput<P> {
    const SIZE: usize = P::SIZE + u64::SIZE;

    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe {
            self.param.encode_into(&mut buffer[..P::SIZE]);
            self.repetition_index
                .encode_into(&mut buffer[P::SIZE..Self::SIZE]);
        }
    }
}

/// Lazy ordered input source for repeated stochastic executions.
///
/// This source represents the logical Cartesian product of `params` and
/// `0..repetitions` without materializing every [`StochasticInput`]. Items are emitted in
/// parameter-major order: all repetitions for `params[0]`, then all repetitions for `params[1]`,
/// and so on.
///
/// Converting the source into an indexed parallel iterator panics if the total number of repeated
/// inputs cannot be represented by `usize`.
#[derive(Debug, Clone)]
pub struct RepeatedStochasticInputs<I> {
    params: I,
    repetitions: usize,
}

impl<I> RepeatedStochasticInputs<I> {
    /// Create a lazy repeated stochastic input source.
    pub const fn new(params: I, repetitions: usize) -> Self {
        Self {
            params,
            repetitions,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn stochastic_input_constructors_encode_identically() {
        let from_new = StochasticInput::new(123u64, 9);
        let from_tuple: StochasticInput<u64> = (123u64, 9u64).into();
        let from_fields = StochasticInput {
            param: 123u64,
            repetition_index: 9,
        };

        let mut buffer_new = vec![0u8; StochasticInput::<u64>::SIZE];
        let mut buffer_tuple = vec![0u8; StochasticInput::<u64>::SIZE];
        let mut buffer_fields = vec![0u8; StochasticInput::<u64>::SIZE];

        let encoded_new = unsafe { from_new.encode_with_buffer(&mut buffer_new) };
        let encoded_tuple = unsafe { from_tuple.encode_with_buffer(&mut buffer_tuple) };
        let encoded_fields = unsafe { from_fields.encode_with_buffer(&mut buffer_fields) };

        assert_eq!(encoded_new, encoded_tuple);
        assert_eq!(encoded_new, encoded_fields);
    }
}
