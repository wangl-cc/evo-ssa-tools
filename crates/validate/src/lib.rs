#![doc = include_str!("../README.md")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

/// Error returned when a parameter fails validation.
///
/// The full diagnostic is available through [`std::fmt::Display`]. Use [`ValidationError::name`]
/// when you need the failing parameter name separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    name: &'static str,
    value: String,
    expected: &'static str,
}

impl ValidationError {
    /// Return the name of the parameter that failed validation.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[doc(hidden)]
    pub fn __new(name: &'static str, value: String, expected: &'static str) -> Self {
        Self {
            name,
            value,
            expected,
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid parameter `{}`: expected {}, got {}",
            self.name, self.expected, self.value
        )
    }
}

impl std::error::Error for ValidationError {}

/// Validate a parameter and return [`ValidationError`] on failure.
///
/// # Examples
///
/// ```
/// use validate::{ValidationError, validate};
///
/// fn configure(rate: f64, probability: f64) -> Result<(), ValidationError> {
///     validate!(rate: > 0.0)?;
///     validate!(probability: >= 0.0, <= 1.0)?;
///     validate!(rate: rate.is_finite(); "finite")?;
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! validate {
    // Range checks use comma-separated clauses and render the corresponding interval.
    // `validate!(config.probability: >= lower(), < upper())`.
    ($value:ident $(.$field:tt)* : $lower_operator:tt $lower:expr, $upper_operator:tt $upper:expr $(,)?) => {
        $crate::validate!(@range (
            stringify!($value $(.$field)*),
            $value $(.$field)*
        ), $lower_operator, $lower, $upper_operator, $upper)
    };
    // Single comparison checks. Bounds are full Rust expressions.
    // `validate!(config.steps: >= min_steps())`.
    ($value:ident $(.$field:tt)* : $operator:tt $bound:expr $(,)?) => {
        $crate::validate!(@cmp (stringify!($value $(.$field)*), $value $(.$field)*), $operator, $bound)
    };
    // Custom predicate that uses the field path text as the diagnostic name:
    // `validate!(sigma: sigma.is_finite(); "finite")`.
    ($value:ident $(.$field:tt)* : $condition:expr ; $expected:expr $(,)?) => {{
        $crate::validate!(@check $condition, stringify!($value $(.$field)*), &$value $(.$field)*, $expected)
    }};
    // Shared comparison implementation. Public comparison arms only extract `(name, value)`;
    // this arm evaluates the value and bound once, then delegates condition checking.
    (@cmp ($name:expr, $value:expr), $operator:tt, $bound:expr) => {{
        let value = &$value;
        let bound = $bound;
        $crate::validate!(
            @check value $operator &bound,
            $name,
            value,
            $crate::validate!(@cmp_expected $operator, $bound)
        )
    }};
    // Shared range implementation. Bounds are evaluated once, and the operators decide whether
    // each side is open or closed.
    (@range (
        $name:expr,
        $value:expr
    ), $lower_operator:tt, $lower:expr, $upper_operator:tt, $upper:expr) => {{
        let value = &$value;
        let lower = $lower;
        let upper = $upper;
        $crate::validate!(
            @check value $lower_operator &lower && value $upper_operator &upper,
            $name,
            value,
            $crate::validate!(@range_expected $lower_operator, $upper_operator, $lower, $upper)
        )
    }};
    (@range_expected $lower_operator:tt, $upper_operator:tt, $lower:expr, $upper:expr) => {
        concat!(
            "in ",
            $crate::validate!(@lower_bracket $lower_operator),
            stringify!($lower),
            ", ",
            stringify!($upper),
            $crate::validate!(@upper_bracket $upper_operator)
        )
    };
    (@lower_bracket >) => {
        "("
    };
    (@lower_bracket >=) => {
        "["
    };
    (@upper_bracket <) => {
        ")"
    };
    (@upper_bracket <=) => {
        "]"
    };
    (@cmp_expected $operator:tt, $bound:expr) => {
        concat!(stringify!($operator), " ", stringify!($bound))
    };
    // Convert a boolean validation result into `Result<(), ValidationError>`.
    (@check $condition:expr, $name:expr, $value:expr, $expected:expr) => {{
        if $condition {
            ::core::result::Result::Ok(())
        } else {
            $crate::validate!(@invalid $name, $value, $expected)
        }
    }};
    // Build the diagnostic error. The value is formatted only on the failure path.
    (@invalid $name:expr, $value:expr, $expected:expr) => {
        ::core::result::Result::Err($crate::ValidationError::__new(
            $name,
            ::std::format!("{:?}", $value),
            $expected,
        ))
    };
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn assert_validation_error(error: ValidationError, name: &'static str, display: &str) {
        assert_eq!(error.name(), name);
        assert_eq!(error.to_string(), display);
    }

    #[test]
    fn display_orders_expected_before_actual_value() {
        let error = ValidationError::__new("rate", "-1.0".to_owned(), "> 0.0");

        assert_eq!(
            error.to_string(),
            "invalid parameter `rate`: expected > 0.0, got -1.0"
        );
    }

    #[test]
    fn comparison_checks_identifier_parameters() {
        let rate = -0.1;
        let error = validate!(rate: > 0.0).unwrap_err();

        assert_validation_error(
            error,
            "rate",
            "invalid parameter `rate`: expected > 0.0, got -0.1",
        );

        let rate = 0.1;
        assert!(validate!(rate: > 0.0).is_ok());
    }

    #[test]
    fn comparison_checks_field_paths() {
        struct Inner {
            steps: usize,
        }

        struct Config {
            inner: Inner,
        }

        let config = Config {
            inner: Inner { steps: 0 },
        };
        let error = validate!(config.inner.steps: >= 1).unwrap_err();

        assert_validation_error(
            error,
            "config.inner.steps",
            "invalid parameter `config.inner.steps`: expected >= 1, got 0",
        );
    }

    #[test]
    fn comparison_checks_tuple_field_paths() {
        let config = (0,);
        let error = validate!(config.0: >= 1).unwrap_err();

        assert_validation_error(
            error,
            "config.0",
            "invalid parameter `config.0`: expected >= 1, got 0",
        );
    }

    #[test]
    fn comparison_checks_self_fields() {
        struct Config {
            rate: f64,
        }

        impl Config {
            fn validate(&self) -> Result<(), ValidationError> {
                validate!(self.rate: > 0.0)
            }
        }

        let error = Config { rate: 0.0 }.validate().unwrap_err();

        assert_validation_error(
            error,
            "self.rate",
            "invalid parameter `self.rate`: expected > 0.0, got 0.0",
        );
    }

    #[test]
    fn comparison_checks_render_each_supported_operator() {
        let count = 3;
        assert_validation_error(
            validate!(count: < 3).unwrap_err(),
            "count",
            "invalid parameter `count`: expected < 3, got 3",
        );
        assert_validation_error(
            validate!(count: <= 2).unwrap_err(),
            "count",
            "invalid parameter `count`: expected <= 2, got 3",
        );

        let count = 0;
        assert_validation_error(
            validate!(count: > 0).unwrap_err(),
            "count",
            "invalid parameter `count`: expected > 0, got 0",
        );
        assert_validation_error(
            validate!(count: >= 1).unwrap_err(),
            "count",
            "invalid parameter `count`: expected >= 1, got 0",
        );
    }

    #[test]
    fn comparison_accepts_expression_bounds() {
        let min_count = 1;
        let count = 1;
        let error = validate!(count: > min_count + 1).unwrap_err();

        assert_validation_error(
            error,
            "count",
            "invalid parameter `count`: expected > min_count + 1, got 1",
        );
    }

    #[test]
    fn inclusive_range_checks_identifier_parameters() {
        let probability = 1.2;
        let error = validate!(probability: >= 0.0, <= 1.0).unwrap_err();

        assert_validation_error(
            error,
            "probability",
            "invalid parameter `probability`: expected in [0.0, 1.0], got 1.2",
        );

        let probability = 0.5;
        assert!(validate!(probability: >= 0.0, <= 1.0).is_ok());
    }

    #[test]
    fn range_checks_render_open_and_half_open_intervals() {
        let probability = 0.0;
        assert_validation_error(
            validate!(probability: > 0.0, < 1.0).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in (0.0, 1.0), got 0.0",
        );
        assert_validation_error(
            validate!(probability: > 0.0, <= 1.0).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in (0.0, 1.0], got 0.0",
        );

        let probability = 1.0;
        assert_validation_error(
            validate!(probability: >= 0.0, < 1.0).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in [0.0, 1.0), got 1.0",
        );
        assert_validation_error(
            validate!(probability: >= 0.0, <= 0.5).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in [0.0, 0.5], got 1.0",
        );
    }

    #[test]
    fn range_checks_support_fields_and_negative_literal_bounds() {
        struct Config {
            score: f64,
        }

        let config = Config { score: -1.0 };
        let error = validate!(config.score: > -1.0, <= 1.0).unwrap_err();

        assert_validation_error(
            error,
            "config.score",
            "invalid parameter `config.score`: expected in (-1.0, 1.0], got -1.0",
        );
    }

    #[test]
    fn range_checks_tuple_field_paths() {
        let config = (1.5,);
        let error = validate!(config.0: >= 0.0, <= 1.0).unwrap_err();

        assert_validation_error(
            error,
            "config.0",
            "invalid parameter `config.0`: expected in [0.0, 1.0], got 1.5",
        );
    }

    #[test]
    fn range_accepts_expression_bounds() {
        let lower = -2.0;
        let upper = 2.0;
        let score = 1.5;
        let error = validate!(score: > lower + 1.0, < upper - 1.0).unwrap_err();

        assert_validation_error(
            error,
            "score",
            "invalid parameter `score`: expected in (lower + 1.0, upper - 1.0), got 1.5",
        );
    }

    #[test]
    fn range_evaluates_expression_bounds_once() {
        fn next(calls: &std::cell::Cell<usize>) -> f64 {
            let current = calls.get();
            calls.set(current + 1);
            current as f64
        }

        let calls = std::cell::Cell::new(0);
        let score = 2.0;
        let error = validate!(score: > next(&calls), < next(&calls)).unwrap_err();

        assert_eq!(calls.get(), 2);
        assert_validation_error(
            error,
            "score",
            "invalid parameter `score`: expected in (next(&calls), next(&calls)), got 2.0",
        );
    }

    #[test]
    fn custom_checks_use_inferred_value_name() {
        let sigma = f64::NAN;
        let error = validate!(sigma: sigma.is_finite(); "finite").unwrap_err();

        assert_validation_error(
            error,
            "sigma",
            "invalid parameter `sigma`: expected finite, got NaN",
        );
    }

    #[test]
    fn custom_checks_report_field_path_value() {
        struct Inner {
            sigma: f64,
        }

        struct Config {
            inner: Inner,
        }

        let config = Config {
            inner: Inner { sigma: f64::NAN },
        };
        let error =
            validate!(config.inner.sigma: config.inner.sigma.is_finite(); "finite").unwrap_err();

        assert_validation_error(
            error,
            "config.inner.sigma",
            "invalid parameter `config.inner.sigma`: expected finite, got NaN",
        );
    }
}
