/// APoisson distribution sampler using Knuth's algorithm.
///
/// This is only suitable for small lambda values (lambda < 30.0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoissonKnuth {
    exp_lambda: f64,
}

impl PoissonKnuth {
    /// Create a new Poisson distribution with the given lambda.
    ///
    /// Returns None if lambda is not positive or too large (lambda > 30.0).
    pub fn new(lambda: f64) -> Option<Self> {
        if 0.0 < lambda && lambda < 30.0 {
            let exp_lambda = (-lambda).exp();
            Some(Self { exp_lambda })
        } else {
            None
        }
    }
}

// As lambda is small, so use u16 as the return type is fine.
impl rand::prelude::Distribution<u16> for PoissonKnuth {
    fn sample<G: rand::Rng + ?Sized>(&self, rng: &mut G) -> u16 {
        // Knuth algorithm
        let mut k = 0;
        let mut p = 1.0;
        while p > self.exp_lambda {
            k += 1;
            p *= rng.random::<f64>();
        }
        k - 1
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, distr::Distribution, rngs::SmallRng};

    use super::*;

    #[test]
    fn test_poisson_new() {
        assert!(PoissonKnuth::new(1.5).is_some());
        assert!(PoissonKnuth::new(0.0).is_none());
        assert!(PoissonKnuth::new(-1.0).is_none());
        assert!(PoissonKnuth::new(40.0).is_none());
    }

    const SAMPLE_SIZE: usize = 10000;

    #[test]
    fn test_poisson_distribution_stats() {
        // For Poisson distribution, mean equals lambda
        let lambda = 5.0;
        let poisson = PoissonKnuth::new(lambda).unwrap();
        let rng = SmallRng::from_os_rng();

        let samples = poisson
            .sample_iter(rng)
            .take(SAMPLE_SIZE)
            .map(|x| x as u64)
            .collect::<Vec<_>>();

        let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
        assert!((mean - lambda).abs() < 0.2); // Tolerance 0.1

        let second_moment =
            samples.iter().map(|&x| x.pow(2)).sum::<u64>() as f64 / samples.len() as f64;
        let variance = second_moment - mean.powi(2);
        assert!((variance - lambda).abs() < 0.2); // Tolerance 0.2
    }
}
