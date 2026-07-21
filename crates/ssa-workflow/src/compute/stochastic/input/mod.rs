//! Stochastic input identity and lazy repeated input sources.

mod parallel;
mod serial;

pub use parallel::RepeatedStochasticIntoParIter;
pub use serial::RepeatedStochasticIntoIter;

use crate::cache::{CanonicalEncode, CanonicalWriter};

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

    fn encode(&self, writer: &mut CanonicalWriter<'_>) {
        writer.write(&self.param).write(&self.repetition_index);
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
    use crate::cache::CanonicalBuffer;

    #[test]
    fn stochastic_input_constructors_encode_identically() {
        let from_new = StochasticInput::new(123u64, 9);
        let from_tuple: StochasticInput<u64> = (123u64, 9u64).into();
        let from_fields = StochasticInput {
            param: 123u64,
            repetition_index: 9,
        };

        let mut buffer_new = CanonicalBuffer::new();
        let mut buffer_tuple = CanonicalBuffer::new();
        let mut buffer_fields = CanonicalBuffer::new();

        let encoded_new = buffer_new.encode(&from_new);
        let encoded_tuple = buffer_tuple.encode(&from_tuple);
        let encoded_fields = buffer_fields.encode(&from_fields);

        assert_eq!(encoded_new, encoded_tuple);
        assert_eq!(encoded_new, encoded_fields);
    }

    fn encode<T: CanonicalEncode>(value: &T) -> Vec<u8> {
        let mut buffer = CanonicalBuffer::<T>::new();
        buffer.encode(value).to_vec()
    }

    #[test]
    fn stochastic_input_inherits_order_from_signed_param_and_repetition() {
        // Parameter-major order: all repetitions for param[0], then param[1], etc.
        let inputs: Vec<StochasticInput<i32>> = vec![
            StochasticInput::new(i32::MIN, 0),
            StochasticInput::new(i32::MIN, 1),
            StochasticInput::new(-1, 0),
            StochasticInput::new(-1, u64::MAX),
            StochasticInput::new(0, 0),
            StochasticInput::new(0, 1),
            StochasticInput::new(1, 0),
            StochasticInput::new(i32::MAX, u64::MAX),
        ];
        let encoded: Vec<Vec<u8>> = inputs.iter().map(encode).collect();
        for (i, window) in encoded.windows(2).enumerate() {
            assert!(
                window[0] < window[1],
                "StochasticInput order violated at index {i}: {:?} !< {:?}",
                inputs[i],
                inputs[i + 1]
            );
        }
    }

    #[test]
    fn stochastic_input_inherits_order_from_float_param() {
        let inputs: Vec<StochasticInput<f64>> = vec![
            StochasticInput::new(f64::NEG_INFINITY, 0),
            StochasticInput::new(-1.5, 0),
            StochasticInput::new(-1.5, 1),
            StochasticInput::new(0.0, 0),
            StochasticInput::new(1.5, 0),
            StochasticInput::new(f64::INFINITY, 0),
            StochasticInput::new(f64::NAN, 0),
        ];
        let encoded: Vec<Vec<u8>> = inputs.iter().map(encode).collect();
        for (i, window) in encoded.windows(2).enumerate() {
            assert!(
                window[0] < window[1],
                "StochasticInput<f64> order violated at index {i}"
            );
        }
    }
}
