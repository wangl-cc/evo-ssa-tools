use super::{RepeatedStochasticInputs, StochasticInput};

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn repeated_inputs_are_parameter_major() {
        let actual = RepeatedStochasticInputs::new(10u32..12, 2)
            .into_iter()
            .collect::<Vec<_>>();
        let expected = vec![
            StochasticInput::new(10, 0),
            StochasticInput::new(10, 1),
            StochasticInput::new(11, 0),
            StochasticInput::new(11, 1),
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn zero_repetitions_are_empty_without_consuming_parameters() {
        let iter = RepeatedStochasticInputs::new(0..usize::MAX, 0).into_iter();

        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.collect::<Vec<_>>(), []);
    }
}
