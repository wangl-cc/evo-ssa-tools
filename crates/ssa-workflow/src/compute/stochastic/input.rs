//! Stochastic input identity and lazy repeated input sources.

use std::ops::Range;

use rayon::{
    iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge},
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
};

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

impl<I> IntoIterator for RepeatedStochasticInputs<I>
where
    I: IntoIterator,
    I::Item: Clone,
{
    type IntoIter = RepeatedStochasticIntoIter<I::IntoIter>;
    type Item = StochasticInput<I::Item>;

    fn into_iter(self) -> Self::IntoIter {
        RepeatedStochasticIntoIter {
            params: self.params.into_iter(),
            current_param: None,
            repetitions: self.repetitions,
            repetition_index: 0,
        }
    }
}

/// Serial iterator over a [`RepeatedStochasticInputs`] source.
#[derive(Debug, Clone)]
pub struct RepeatedStochasticIntoIter<I>
where
    I: Iterator,
{
    params: I,
    current_param: Option<I::Item>,
    repetitions: usize,
    repetition_index: usize,
}

impl<I> Iterator for RepeatedStochasticIntoIter<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = StochasticInput<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.repetitions == 0 {
            return None;
        }

        if self.current_param.is_none() {
            self.current_param = self.params.next();
            self.repetition_index = 0;
        }

        let param = self.current_param.as_ref()?.clone();
        let repetition_index = self.repetition_index;
        self.repetition_index += 1;

        if self.repetition_index == self.repetitions {
            self.current_param = None;
        }

        Some(StochasticInput::new(param, repetition_index as u64))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.repetitions == 0 {
            return (0, Some(0));
        }

        let current_remaining = self
            .current_param
            .as_ref()
            .map_or(0, |_| self.repetitions - self.repetition_index);
        let (lower, upper) = self.params.size_hint();
        let lower = lower
            .saturating_mul(self.repetitions)
            .saturating_add(current_remaining);
        let upper = upper
            .and_then(|value| value.checked_mul(self.repetitions))
            .and_then(|value| value.checked_add(current_remaining));
        (lower, upper)
    }
}

impl<I> IntoParallelIterator for RepeatedStochasticInputs<I>
where
    I: IntoParallelIterator,
    I::Item: Clone + Send,
    I::Iter: IndexedParallelIterator,
{
    type Item = StochasticInput<I::Item>;
    type Iter = RepeatedStochasticIntoParIter<I::Iter>;

    fn into_par_iter(self) -> Self::Iter {
        let params = self.params.into_par_iter();
        let parameter_count = params.len();
        let Some(len) = parameter_count.checked_mul(self.repetitions) else {
            panic!(
                "repeated stochastic input count overflow: {parameter_count} parameters with {} repetitions",
                self.repetitions,
            );
        };

        RepeatedStochasticIntoParIter {
            params,
            repetitions: self.repetitions,
            len,
        }
    }
}

/// Indexed parallel iterator over a [`RepeatedStochasticInputs`] source.
#[derive(Debug, Clone)]
pub struct RepeatedStochasticIntoParIter<I> {
    params: I,
    repetitions: usize,
    len: usize,
}

impl<I> ParallelIterator for RepeatedStochasticIntoParIter<I>
where
    I: IndexedParallelIterator,
    I::Item: Clone,
{
    type Item = StochasticInput<I::Item>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len)
    }
}

impl<I> IndexedParallelIterator for RepeatedStochasticIntoParIter<I>
where
    I: IndexedParallelIterator,
    I::Item: Clone,
{
    fn len(&self) -> usize {
        self.len
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        let parameter_count = self.params.len();
        self.params.with_producer(RepeatedStochasticCallback {
            callback,
            parameter_count,
            repetitions: self.repetitions,
        })
    }
}

struct RepeatedStochasticCallback<CB> {
    callback: CB,
    parameter_count: usize,
    repetitions: usize,
}

impl<P, CB> ProducerCallback<P> for RepeatedStochasticCallback<CB>
where
    P: Clone + Send,
    CB: ProducerCallback<StochasticInput<P>>,
{
    type Output = CB::Output;

    fn callback<B>(self, base: B) -> Self::Output
    where
        B: Producer<Item = P>,
    {
        self.callback.callback(RepeatedStochasticProducer {
            base,
            parameter_count: self.parameter_count,
            repetitions: self.repetitions,
            prefix: None,
            suffix: None,
        })
    }
}

struct RepeatedStochasticProducer<B>
where
    B: Producer,
{
    base: B,
    parameter_count: usize,
    repetitions: usize,
    prefix: Option<RepeatedParameter<B::Item>>,
    suffix: Option<RepeatedParameter<B::Item>>,
}

impl<B> RepeatedStochasticProducer<B>
where
    B: Producer,
    B::Item: Clone,
{
    fn len(&self) -> usize {
        self.prefix.as_ref().map_or(0, RepeatedParameter::len)
            + self.parameter_count * self.repetitions
            + self.suffix.as_ref().map_or(0, RepeatedParameter::len)
    }

    fn split_zero_repetitions(self) -> (Self, Self) {
        debug_assert_eq!(self.len(), 0);
        let (left, right) = self.base.split_at(0);
        (
            Self {
                base: left,
                parameter_count: 0,
                repetitions: 0,
                prefix: None,
                suffix: None,
            },
            Self {
                base: right,
                parameter_count: self.parameter_count,
                repetitions: 0,
                prefix: None,
                suffix: None,
            },
        )
    }
}

impl<B> Producer for RepeatedStochasticProducer<B>
where
    B: Producer,
    B::Item: Clone + Send,
{
    type IntoIter = RepeatedStochasticProducerIter<B::IntoIter>;
    type Item = StochasticInput<B::Item>;

    fn into_iter(self) -> Self::IntoIter {
        RepeatedStochasticProducerIter {
            prefix: self.prefix,
            front: None,
            base: self.base.into_iter(),
            back: None,
            suffix: self.suffix,
            repetitions: self.repetitions,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len());
        if self.repetitions == 0 {
            assert_eq!(index, 0);
            return self.split_zero_repetitions();
        }

        let prefix_len = self.prefix.as_ref().map_or(0, RepeatedParameter::len);
        if index < prefix_len {
            let (left_prefix, right_prefix) = self
                .prefix
                .expect("prefix length is non-zero")
                .split_at(index);
            let (left_base, right_base) = self.base.split_at(0);
            return (
                Self {
                    base: left_base,
                    parameter_count: 0,
                    repetitions: self.repetitions,
                    prefix: left_prefix,
                    suffix: None,
                },
                Self {
                    base: right_base,
                    parameter_count: self.parameter_count,
                    repetitions: self.repetitions,
                    prefix: right_prefix,
                    suffix: self.suffix,
                },
            );
        }

        let base_index = index - prefix_len;
        let base_len = self.parameter_count * self.repetitions;
        if base_index <= base_len {
            let parameter_index = base_index / self.repetitions;
            let repetition_index = base_index % self.repetitions;
            let (left_base, right_base) = self.base.split_at(parameter_index);

            if repetition_index == 0 {
                return (
                    Self {
                        base: left_base,
                        parameter_count: parameter_index,
                        repetitions: self.repetitions,
                        prefix: self.prefix,
                        suffix: None,
                    },
                    Self {
                        base: right_base,
                        parameter_count: self.parameter_count - parameter_index,
                        repetitions: self.repetitions,
                        prefix: None,
                        suffix: self.suffix,
                    },
                );
            }

            let (boundary, right_base) = right_base.split_at(1);
            let mut boundary = boundary.into_iter();
            let parameter = boundary
                .next()
                .expect("a producer split at one must contain one parameter");
            debug_assert!(boundary.next().is_none());
            let (left_suffix, right_prefix) =
                RepeatedParameter::new(parameter, 0..self.repetitions).split_at(repetition_index);

            return (
                Self {
                    base: left_base,
                    parameter_count: parameter_index,
                    repetitions: self.repetitions,
                    prefix: self.prefix,
                    suffix: left_suffix,
                },
                Self {
                    base: right_base,
                    parameter_count: self.parameter_count - parameter_index - 1,
                    repetitions: self.repetitions,
                    prefix: right_prefix,
                    suffix: self.suffix,
                },
            );
        }

        let suffix_index = base_index - base_len;
        let (left_suffix, right_prefix) = self
            .suffix
            .expect("the split index lies within the suffix")
            .split_at(suffix_index);
        let (left_base, right_base) = self.base.split_at(self.parameter_count);

        (
            Self {
                base: left_base,
                parameter_count: self.parameter_count,
                repetitions: self.repetitions,
                prefix: self.prefix,
                suffix: left_suffix,
            },
            Self {
                base: right_base,
                parameter_count: 0,
                repetitions: self.repetitions,
                prefix: right_prefix,
                suffix: None,
            },
        )
    }
}

#[derive(Debug, Clone)]
struct RepeatedParameter<P> {
    parameter: P,
    range: Range<usize>,
}

impl<P> RepeatedParameter<P>
where
    P: Clone,
{
    fn new(parameter: P, range: Range<usize>) -> Self {
        Self { parameter, range }
    }

    fn len(&self) -> usize {
        self.range.len()
    }

    fn split_at(self, index: usize) -> (Option<Self>, Option<Self>) {
        assert!(index <= self.len());
        if index == 0 {
            return (None, Some(self));
        }
        if index == self.len() {
            return (Some(self), None);
        }

        let middle = self.range.start + index;
        let right_parameter = self.parameter.clone();
        (
            Some(Self::new(self.parameter, self.range.start..middle)),
            Some(Self::new(right_parameter, middle..self.range.end)),
        )
    }
}

impl<P> Iterator for RepeatedParameter<P>
where
    P: Clone,
{
    type Item = StochasticInput<P>;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|repetition_index| {
            StochasticInput::new(self.parameter.clone(), repetition_index as u64)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<P> DoubleEndedIterator for RepeatedParameter<P>
where
    P: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|repetition_index| {
            StochasticInput::new(self.parameter.clone(), repetition_index as u64)
        })
    }
}

impl<P> ExactSizeIterator for RepeatedParameter<P> where P: Clone {}

struct RepeatedStochasticProducerIter<I>
where
    I: Iterator,
{
    prefix: Option<RepeatedParameter<I::Item>>,
    front: Option<RepeatedParameter<I::Item>>,
    base: I,
    back: Option<RepeatedParameter<I::Item>>,
    suffix: Option<RepeatedParameter<I::Item>>,
    repetitions: usize,
}

impl<I> RepeatedStochasticProducerIter<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn next_from(
        segment: &mut Option<RepeatedParameter<I::Item>>,
    ) -> Option<StochasticInput<I::Item>> {
        let item = segment.as_mut()?.next();
        if segment.as_ref().is_some_and(|segment| segment.len() == 0) {
            *segment = None;
        }
        item
    }

    fn next_back_from(
        segment: &mut Option<RepeatedParameter<I::Item>>,
    ) -> Option<StochasticInput<I::Item>> {
        let item = segment.as_mut()?.next_back();
        if segment.as_ref().is_some_and(|segment| segment.len() == 0) {
            *segment = None;
        }
        item
    }
}

impl<I> Iterator for RepeatedStochasticProducerIter<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
    I::Item: Clone,
{
    type Item = StochasticInput<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.repetitions == 0 {
            return None;
        }
        if let Some(item) = Self::next_from(&mut self.prefix) {
            return Some(item);
        }
        if let Some(item) = Self::next_from(&mut self.front) {
            return Some(item);
        }
        if let Some(parameter) = self.base.next() {
            self.front = Some(RepeatedParameter::new(parameter, 0..self.repetitions));
            return Self::next_from(&mut self.front);
        }
        if let Some(item) = Self::next_from(&mut self.back) {
            return Some(item);
        }
        Self::next_from(&mut self.suffix)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<I> DoubleEndedIterator for RepeatedStochasticProducerIter<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
    I::Item: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.repetitions == 0 {
            return None;
        }
        if let Some(item) = Self::next_back_from(&mut self.suffix) {
            return Some(item);
        }
        if let Some(item) = Self::next_back_from(&mut self.back) {
            return Some(item);
        }
        if let Some(parameter) = self.base.next_back() {
            self.back = Some(RepeatedParameter::new(parameter, 0..self.repetitions));
            return Self::next_back_from(&mut self.back);
        }
        if let Some(item) = Self::next_back_from(&mut self.front) {
            return Some(item);
        }
        Self::next_back_from(&mut self.prefix)
    }
}

impl<I> ExactSizeIterator for RepeatedStochasticProducerIter<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
    I::Item: Clone,
{
    fn len(&self) -> usize {
        self.prefix.as_ref().map_or(0, RepeatedParameter::len)
            + self.front.as_ref().map_or(0, RepeatedParameter::len)
            + self.base.len() * self.repetitions
            + self.back.as_ref().map_or(0, RepeatedParameter::len)
            + self.suffix.as_ref().map_or(0, RepeatedParameter::len)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    struct DualModeInputs {
        values: Range<u32>,
        serial_conversions: Arc<AtomicUsize>,
        parallel_conversions: Arc<AtomicUsize>,
    }

    impl IntoIterator for DualModeInputs {
        type IntoIter = Range<u32>;
        type Item = u32;

        fn into_iter(self) -> Self::IntoIter {
            self.serial_conversions.fetch_add(1, Ordering::SeqCst);
            self.values
        }
    }

    impl IntoParallelIterator for DualModeInputs {
        type Item = u32;
        type Iter = <Range<u32> as IntoParallelIterator>::Iter;

        fn into_par_iter(self) -> Self::Iter {
            self.parallel_conversions.fetch_add(1, Ordering::SeqCst);
            self.values.into_par_iter()
        }
    }

    enum SplitPattern {
        Prefix,
        Suffix,
    }

    struct CollectSplitPattern(SplitPattern);

    impl ProducerCallback<StochasticInput<u32>> for CollectSplitPattern {
        type Output = Vec<StochasticInput<u32>>;

        fn callback<P>(self, producer: P) -> Self::Output
        where
            P: Producer<Item = StochasticInput<u32>>,
        {
            let (first, second, third) = match self.0 {
                SplitPattern::Prefix => {
                    let (first, remainder) = producer.split_at(1);
                    let (second, third) = remainder.split_at(1);
                    (first, second, third)
                }
                SplitPattern::Suffix => {
                    let (remainder, third) = producer.split_at(5);
                    let (first, second) = remainder.split_at(4);
                    (first, second, third)
                }
            };

            first
                .into_iter()
                .chain(second.into_iter())
                .chain(third.into_iter())
                .collect()
        }
    }

    struct CollectMixedEnds;

    impl ProducerCallback<StochasticInput<u32>> for CollectMixedEnds {
        type Output = Vec<StochasticInput<u32>>;

        fn callback<P>(self, producer: P) -> Self::Output
        where
            P: Producer<Item = StochasticInput<u32>>,
        {
            let mut iter = producer.into_iter();
            assert_eq!(iter.size_hint(), (6, Some(6)));
            let actual = vec![
                iter.next().expect("the producer has a front item"),
                iter.next_back().expect("the producer has a back item"),
                iter.next().expect("the front parameter has another repeat"),
                iter.next_back()
                    .expect("the back parameter has another repeat"),
                iter.next().expect("the front parameter has a final repeat"),
                iter.next_back()
                    .expect("the back parameter has a final repeat"),
            ];
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
            actual
        }
    }

    enum CrossedEnd {
        Front,
        Back,
    }

    struct CollectCrossedEnd(CrossedEnd);

    impl ProducerCallback<StochasticInput<u32>> for CollectCrossedEnd {
        type Output = Vec<StochasticInput<u32>>;

        fn callback<P>(self, producer: P) -> Self::Output
        where
            P: Producer<Item = StochasticInput<u32>>,
        {
            let mut iter = producer.into_iter();
            match self.0 {
                CrossedEnd::Front => vec![
                    iter.next().expect("the producer has a front item"),
                    iter.next_back().expect("the producer has a back item"),
                    iter.next_back()
                        .expect("the back parameter has another repeat"),
                    iter.next_back()
                        .expect("the back parameter has a final repeat"),
                    iter.next_back()
                        .expect("the front parameter remains available from the back"),
                    iter.next_back()
                        .expect("the front parameter has one remaining repeat"),
                ],
                CrossedEnd::Back => vec![
                    iter.next_back().expect("the producer has a back item"),
                    iter.next().expect("the producer has a front item"),
                    iter.next().expect("the front parameter has another repeat"),
                    iter.next().expect("the front parameter has a final repeat"),
                    iter.next()
                        .expect("the back parameter remains available from the front"),
                    iter.next()
                        .expect("the back parameter has one remaining repeat"),
                ],
            }
        }
    }

    struct SplitEmpty;

    impl ProducerCallback<StochasticInput<u32>> for SplitEmpty {
        type Output = (usize, usize);

        fn callback<P>(self, producer: P) -> Self::Output
        where
            P: Producer<Item = StochasticInput<u32>>,
        {
            let (left, right) = producer.split_at(0);
            let mut left = left.into_iter();
            let mut right = right.into_iter();
            assert_eq!(left.next_back(), None);
            assert_eq!(right.next_back(), None);
            (left.len(), right.len())
        }
    }

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

    #[test]
    fn repeated_inputs_support_standard_serial_and_parallel_iteration() {
        let inputs = RepeatedStochasticInputs::new(10u32..12, 2);
        let expected = vec![
            StochasticInput::new(10, 0),
            StochasticInput::new(10, 1),
            StochasticInput::new(11, 0),
            StochasticInput::new(11, 1),
        ];
        let expected_len = expected.len();

        assert_eq!(inputs.clone().into_iter().collect::<Vec<_>>(), expected);

        let parallel_inputs = inputs.clone().into_par_iter();
        assert_eq!(parallel_inputs.opt_len(), Some(expected_len));
        assert_eq!(parallel_inputs.len(), expected_len);

        let mut parallel_outputs = Vec::new();
        parallel_inputs.collect_into_vec(&mut parallel_outputs);
        assert_eq!(parallel_outputs, expected);

        let indexed_outputs = inputs
            .into_par_iter()
            .zip(0..expected_len)
            .collect::<Vec<_>>();
        let expected_indexed = expected
            .into_iter()
            .zip(0..expected_len)
            .collect::<Vec<_>>();
        assert_eq!(indexed_outputs, expected_indexed);
    }

    #[test]
    fn repeated_inputs_use_the_selected_source_mode_directly() {
        let serial_conversions = Arc::new(AtomicUsize::new(0));
        let parallel_conversions = Arc::new(AtomicUsize::new(0));
        let inputs = DualModeInputs {
            values: 10..12,
            serial_conversions: Arc::clone(&serial_conversions),
            parallel_conversions: Arc::clone(&parallel_conversions),
        };
        let serial = RepeatedStochasticInputs::new(inputs, 2)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(serial.len(), 4);
        assert_eq!(serial_conversions.load(Ordering::SeqCst), 1);
        assert_eq!(parallel_conversions.load(Ordering::SeqCst), 0);

        let inputs = DualModeInputs {
            values: 10..12,
            serial_conversions: Arc::clone(&serial_conversions),
            parallel_conversions: Arc::clone(&parallel_conversions),
        };
        let parallel = RepeatedStochasticInputs::new(inputs, 2)
            .into_par_iter()
            .collect::<Vec<_>>();

        assert_eq!(parallel.len(), 4);
        assert_eq!(serial_conversions.load(Ordering::SeqCst), 1);
        assert_eq!(parallel_conversions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn repeated_parallel_producer_splits_at_every_output_boundary() {
        let params = [10u32, 20];
        let expected = vec![
            StochasticInput::new(10, 0),
            StochasticInput::new(10, 1),
            StochasticInput::new(10, 2),
            StochasticInput::new(20, 0),
            StochasticInput::new(20, 1),
            StochasticInput::new(20, 2),
        ];

        for split_index in 0..=expected.len() {
            let left = RepeatedStochasticInputs::new(params, 3)
                .into_par_iter()
                .take(split_index)
                .collect::<Vec<_>>();
            let right = RepeatedStochasticInputs::new(params, 3)
                .into_par_iter()
                .skip(split_index)
                .collect::<Vec<_>>();
            let actual = left.into_iter().chain(right).collect::<Vec<_>>();

            assert_eq!(actual, expected, "split index {split_index}");
        }

        let reversed = RepeatedStochasticInputs::new(params, 3)
            .into_par_iter()
            .rev()
            .collect::<Vec<_>>();
        assert_eq!(reversed, expected.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn repeated_parallel_producer_supports_recursive_boundary_splits() {
        let expected = RepeatedStochasticInputs::new([10u32, 20], 3)
            .into_iter()
            .collect::<Vec<_>>();
        let prefix_split = RepeatedStochasticInputs::new([10u32, 20], 3)
            .into_par_iter()
            .with_producer(CollectSplitPattern(SplitPattern::Prefix));
        let suffix_split = RepeatedStochasticInputs::new([10u32, 20], 3)
            .into_par_iter()
            .with_producer(CollectSplitPattern(SplitPattern::Suffix));

        assert_eq!(prefix_split, expected);
        assert_eq!(suffix_split, expected);
    }

    #[test]
    fn repeated_parallel_producer_supports_mixed_double_ended_iteration() {
        let actual = RepeatedStochasticInputs::new([10u32, 20], 3)
            .into_par_iter()
            .with_producer(CollectMixedEnds);
        let expected = vec![
            StochasticInput::new(10, 0),
            StochasticInput::new(20, 2),
            StochasticInput::new(10, 1),
            StochasticInput::new(20, 1),
            StochasticInput::new(10, 2),
            StochasticInput::new(20, 0),
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn repeated_parallel_producer_preserves_partially_consumed_ends() {
        let from_front = RepeatedStochasticInputs::new([10u32, 20], 3)
            .into_par_iter()
            .with_producer(CollectCrossedEnd(CrossedEnd::Front));
        let from_back = RepeatedStochasticInputs::new([10u32, 20], 3)
            .into_par_iter()
            .with_producer(CollectCrossedEnd(CrossedEnd::Back));

        assert_eq!(from_front, vec![
            StochasticInput::new(10, 0),
            StochasticInput::new(20, 2),
            StochasticInput::new(20, 1),
            StochasticInput::new(20, 0),
            StochasticInput::new(10, 2),
            StochasticInput::new(10, 1),
        ]);
        assert_eq!(from_back, vec![
            StochasticInput::new(20, 2),
            StochasticInput::new(10, 0),
            StochasticInput::new(10, 1),
            StochasticInput::new(10, 2),
            StochasticInput::new(20, 0),
            StochasticInput::new(20, 1),
        ]);
    }

    #[test]
    fn repeated_parameter_splits_at_both_edges() {
        let segment = RepeatedParameter::new(10u32, 0..3);
        assert_eq!(segment.size_hint(), (3, Some(3)));
        let (left, right) = segment.split_at(0);
        assert!(left.is_none());
        assert_eq!(right.as_ref().map(RepeatedParameter::len), Some(3));

        let (left, right) = right
            .expect("the zero split preserves the right segment")
            .split_at(3);
        assert_eq!(left.as_ref().map(RepeatedParameter::len), Some(3));
        assert!(right.is_none());
    }

    #[test]
    fn zero_repetitions_are_empty_without_consuming_parameters() {
        let serial = RepeatedStochasticInputs::new(0..usize::MAX, 0).into_iter();
        assert_eq!(serial.size_hint(), (0, Some(0)));
        let serial = serial.collect::<Vec<_>>();
        let parallel_inputs =
            RepeatedStochasticInputs::new(0u32..usize::MAX as u32, 0).into_par_iter();
        let split_lengths = parallel_inputs.with_producer(SplitEmpty);

        assert!(serial.is_empty());
        assert_eq!(split_lengths, (0, 0));
    }

    #[test]
    #[should_panic(expected = "repeated stochastic input count overflow")]
    fn parallel_repeated_inputs_reject_unrepresentable_length() {
        let _ = RepeatedStochasticInputs::new([1u8, 2], usize::MAX).into_par_iter();
    }
}
