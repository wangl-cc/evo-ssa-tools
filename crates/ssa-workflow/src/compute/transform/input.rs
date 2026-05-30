use crate::cache::{CacheSchema, CanonicalEncode, extend_schema_signature, schema_signature};

/// Input for a parameterized dependent transform.
///
/// Canonical payload encoding is `param` followed by the source payload, which groups cache keys by
/// transform parameter for prefix-oriented storage backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependentInput<P, S> {
    /// Parameter for the dependent transform.
    pub param: P,
    /// Input for the upstream source computation.
    pub source: S,
}

impl<P, S> DependentInput<P, S> {
    /// Create dependent input from transform parameter and source input.
    pub const fn new(param: P, source: S) -> Self {
        Self { param, source }
    }
}

unsafe impl<P: CacheSchema, S: CacheSchema> CacheSchema for DependentInput<P, S> {
    const SCHEMA_SIGNATURE: u32 = {
        let signature = schema_signature(b"ssa-workflow:cache-schema:v1;dependent-input");
        let signature = extend_schema_signature(signature, P::SCHEMA_SIGNATURE);
        extend_schema_signature(signature, S::SCHEMA_SIGNATURE)
    };
}

unsafe impl<P: CanonicalEncode, S: CanonicalEncode> CanonicalEncode for DependentInput<P, S> {
    const SIZE: usize = P::SIZE + S::SIZE;

    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe {
            self.param.encode_into(&mut buffer[..P::SIZE]);
            self.source.encode_into(&mut buffer[P::SIZE..Self::SIZE]);
        }
    }
}

/// Input for a stochastic dependent transform.
///
/// Canonical payload encoding is `param | source payload | transform_repetition_index`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependentStochasticInput<P, S> {
    /// Parameter for the stochastic dependent transform.
    pub param: P,
    /// Input for the upstream source computation.
    pub source: S,
    /// Repetition index for the stochastic transform itself.
    pub repetition_index: u64,
}

impl<P, S> DependentStochasticInput<P, S> {
    /// Create stochastic dependent input from transform parameter, source input, and repetition.
    pub const fn new(param: P, source: S, repetition_index: u64) -> Self {
        Self {
            param,
            source,
            repetition_index,
        }
    }
}

impl<S> DependentStochasticInput<(), S> {
    /// Create stochastic dependent input for a transform with no explicit parameter.
    pub const fn from_source(source: S, repetition_index: u64) -> Self {
        Self {
            param: (),
            source,
            repetition_index,
        }
    }
}

unsafe impl<P: CacheSchema, S: CacheSchema> CacheSchema for DependentStochasticInput<P, S> {
    const SCHEMA_SIGNATURE: u32 = {
        let signature =
            schema_signature(b"ssa-workflow:cache-schema:v1;dependent-stochastic-input");
        let signature = extend_schema_signature(signature, P::SCHEMA_SIGNATURE);
        let signature = extend_schema_signature(signature, S::SCHEMA_SIGNATURE);
        extend_schema_signature(signature, u64::SCHEMA_SIGNATURE)
    };
}

unsafe impl<P: CanonicalEncode, S: CanonicalEncode> CanonicalEncode
    for DependentStochasticInput<P, S>
{
    const SIZE: usize = P::SIZE + S::SIZE + u64::SIZE;

    unsafe fn encode_into(&self, buffer: &mut [u8]) {
        unsafe {
            self.param.encode_into(&mut buffer[..P::SIZE]);
            self.source
                .encode_into(&mut buffer[P::SIZE..P::SIZE + S::SIZE]);
            self.repetition_index
                .encode_into(&mut buffer[P::SIZE + S::SIZE..Self::SIZE]);
        }
    }
}
