use rand::{Rng, RngExt};

use crate::{Error, Result};

pub(crate) fn checked_propensity(value: f64) -> Result<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(Error::InvalidPropensity { value })
    }
}

pub(crate) fn draw_exponential<R: Rng + ?Sized>(rate: f64, rng: &mut R) -> f64 {
    if rate <= 0.0 {
        f64::INFINITY
    } else {
        -rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0).ln() / rate
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    #[test]
    fn propensity_validation_accepts_non_negative_finite_values() {
        assert_eq!(checked_propensity(0.0).expect("zero is valid"), 0.0);
        assert_eq!(checked_propensity(2.5).expect("finite rate is valid"), 2.5);
    }

    #[test]
    fn propensity_validation_rejects_invalid_values() {
        assert!(checked_propensity(-1.0).is_err());
        assert!(checked_propensity(f64::INFINITY).is_err());
        assert!(checked_propensity(f64::NAN).is_err());
    }

    #[test]
    fn exponential_draw_handles_active_and_inactive_rates() {
        let mut rng = SmallRng::seed_from_u64(7);
        let wait = draw_exponential(2.0, &mut rng);

        assert!(wait.is_finite() && wait > 0.0);
        assert_eq!(draw_exponential(0.0, &mut rng), f64::INFINITY);
    }
}
