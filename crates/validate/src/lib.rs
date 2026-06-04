#![doc = include_str!("../README.md")]

/// Error returned when a parameter fails validation.
///
/// The full diagnostic is available through [`std::fmt::Display`]. Use [`ParameterError::name`]
/// when you need the failing parameter name separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterError {
    name: &'static str,
    value: String,
    expected: &'static str,
}

impl ParameterError {
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

impl std::fmt::Display for ParameterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid parameter `{}`: expected {}, got {}",
            self.name, self.expected, self.value
        )
    }
}

impl std::error::Error for ParameterError {}

/// Validate a parameter and return [`ParameterError`] on failure.
///
/// # Examples
///
/// ```
/// use validate::{ParameterError, validate};
///
/// fn configure(rate: f64, probability: f64) -> Result<(), ParameterError> {
///     validate!(rate > 0.0)?;
///     validate!(0.0 <= probability <= 1.0)?;
///     validate!(rate, rate.is_finite(), "finite")?;
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! validate {
    // Range with parenthesized expression bounds for a plain identifier:
    // `validate!((min()) <= probability < (max()))`.
    (($lower:expr) $lower_operator:tt $name:ident $upper_operator:tt ($upper:expr) $(,)?) => {
        $crate::validate!(
            @range (stringify!($name), $name),
            $lower,
            $lower_operator,
            $upper_operator,
            $upper
        )
    };
    // Range with parenthesized expression bounds for one field access:
    // `validate!((min()) <= config.probability < (max()))`.
    (($lower:expr) $lower_operator:tt $receiver:tt.$field:ident $upper_operator:tt ($upper:expr) $(,)?) => {
        $crate::validate!(
            @range (stringify!($field), $receiver.$field),
            $lower,
            $lower_operator,
            $upper_operator,
            $upper
        )
    };
    // Range with literal bounds for a plain identifier:
    // `validate!(0.0 <= probability < 1.0)`.
    ($lower:literal $lower_operator:tt $name:ident $upper_operator:tt $upper:literal $(,)?) => {
        $crate::validate!(
            @range (stringify!($name), $name),
            $lower,
            $lower_operator,
            $upper_operator,
            $upper
        )
    };
    // Range with literal bounds for one field access:
    // `validate!(0.0 <= config.probability < 1.0)`.
    ($lower:literal $lower_operator:tt $receiver:tt.$field:ident $upper_operator:tt $upper:literal $(,)?) => {
        $crate::validate!(
            @range (stringify!($field), $receiver.$field),
            $lower,
            $lower_operator,
            $upper_operator,
            $upper
        )
    };
    // Comparison for one field access:
    // `validate!(config.steps >= 1)`.
    ($receiver:tt.$field:ident $operator:tt $bound:expr $(,)?) => {
        $crate::validate!(@cmp (stringify!($field), $receiver.$field), $operator, $bound)
    };
    // Comparison for a plain identifier:
    // `validate!(steps >= 1)`.
    ($name:ident $operator:tt $bound:expr $(,)?) => {
        $crate::validate!(@cmp (stringify!($name), $name), $operator, $bound)
    };
    // Custom predicate with an explicit diagnostic name:
    // `validate!("sigma", value, value.is_finite(), "finite")`.
    ($name:literal, $value:expr, $condition:expr, $expected:expr $(,)?) => {{
        $crate::validate!(@check $condition, $name, &$value, $expected)
    }};
    // Custom predicate that uses the value expression text as the diagnostic name:
    // `validate!(sigma, sigma.is_finite(), "finite")`.
    ($value:expr, $condition:expr, $expected:expr $(,)?) => {{
        $crate::validate!(@check $condition, stringify!($value), &$value, $expected)
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
    // Shared range implementation. Lower and upper are evaluated once, and the operators decide
    // whether each side is open or closed.
    (@range (
        $name:expr,
        $value:expr
    ), $lower:expr, $lower_operator:tt, $upper_operator:tt, $upper:expr) => {{
        let value = &$value;
        let lower = $lower;
        let upper = $upper;
        $crate::validate!(
            @check &lower $lower_operator value && value $upper_operator &upper,
            $name,
            value,
            $crate::validate!(@range_expected $lower_operator, $upper_operator, $lower, $upper)
        )
    }};
    // Render the expected interval text for open and closed range combinations.
    (@range_expected <, <, $lower:expr, $upper:expr) => {
        concat!("in (", stringify!($lower), ", ", stringify!($upper), ")")
    };
    (@range_expected <, <=, $lower:expr, $upper:expr) => {
        concat!("in (", stringify!($lower), ", ", stringify!($upper), "]")
    };
    (@range_expected <=, <, $lower:expr, $upper:expr) => {
        concat!("in [", stringify!($lower), ", ", stringify!($upper), ")")
    };
    (@range_expected <=, <=, $lower:expr, $upper:expr) => {
        concat!("in [", stringify!($lower), ", ", stringify!($upper), "]")
    };
    // Render the expected comparison text for the supported comparison operators.
    (@cmp_expected >, $bound:expr) => {
        concat!("> ", stringify!($bound))
    };
    (@cmp_expected >=, $bound:expr) => {
        concat!(">= ", stringify!($bound))
    };
    (@cmp_expected <, $bound:expr) => {
        concat!("< ", stringify!($bound))
    };
    (@cmp_expected <=, $bound:expr) => {
        concat!("<= ", stringify!($bound))
    };
    // Convert a boolean validation result into `Result<(), ParameterError>`.
    (@check $condition:expr, $name:expr, $value:expr, $expected:expr) => {{
        if $condition {
            ::core::result::Result::Ok(())
        } else {
            $crate::validate!(@invalid $name, $value, $expected)
        }
    }};
    // Build the diagnostic error. The value is formatted only on the failure path.
    (@invalid $name:expr, $value:expr, $expected:expr) => {
        ::core::result::Result::Err($crate::ParameterError::__new(
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

    fn assert_parameter_error(error: ParameterError, name: &'static str, display: &str) {
        assert_eq!(error.name(), name);
        assert_eq!(error.to_string(), display);
    }

    #[test]
    fn display_orders_expected_before_actual_value() {
        let error = ParameterError::__new("rate", "-1.0".to_owned(), "> 0.0");

        assert_eq!(
            error.to_string(),
            "invalid parameter `rate`: expected > 0.0, got -1.0"
        );
    }

    #[test]
    fn comparison_checks_identifier_parameters() {
        let rate = -0.1;
        let error = validate!(rate > 0.0).unwrap_err();

        assert_parameter_error(
            error,
            "rate",
            "invalid parameter `rate`: expected > 0.0, got -0.1",
        );

        let rate = 0.1;
        assert!(validate!(rate > 0.0).is_ok());
    }

    #[test]
    fn comparison_checks_field_parameters() {
        struct Config {
            steps: usize,
        }

        let config = Config { steps: 0 };
        let error = validate!(config.steps >= 1).unwrap_err();

        assert_parameter_error(
            error,
            "steps",
            "invalid parameter `steps`: expected >= 1, got 0",
        );
    }

    #[test]
    fn comparison_checks_self_fields() {
        struct Config {
            rate: f64,
        }

        impl Config {
            fn validate(&self) -> Result<(), ParameterError> {
                validate!(self.rate > 0.0)
            }
        }

        let error = Config { rate: 0.0 }.validate().unwrap_err();

        assert_parameter_error(
            error,
            "rate",
            "invalid parameter `rate`: expected > 0.0, got 0.0",
        );
    }

    #[test]
    fn comparison_checks_render_each_supported_operator() {
        let count = 3;
        assert_parameter_error(
            validate!(count < 3).unwrap_err(),
            "count",
            "invalid parameter `count`: expected < 3, got 3",
        );
        assert_parameter_error(
            validate!(count <= 2).unwrap_err(),
            "count",
            "invalid parameter `count`: expected <= 2, got 3",
        );

        let count = 0;
        assert_parameter_error(
            validate!(count > 0).unwrap_err(),
            "count",
            "invalid parameter `count`: expected > 0, got 0",
        );
        assert_parameter_error(
            validate!(count >= 1).unwrap_err(),
            "count",
            "invalid parameter `count`: expected >= 1, got 0",
        );
    }

    #[test]
    fn inclusive_range_checks_identifier_parameters() {
        let probability = 1.2;
        let error = validate!(0.0 <= probability <= 1.0).unwrap_err();

        assert_parameter_error(
            error,
            "probability",
            "invalid parameter `probability`: expected in [0.0, 1.0], got 1.2",
        );

        let probability = 0.5;
        assert!(validate!(0.0 <= probability <= 1.0).is_ok());
    }

    #[test]
    fn range_checks_render_open_and_half_open_intervals() {
        let probability = 0.0;
        assert_parameter_error(
            validate!(0.0 < probability < 1.0).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in (0.0, 1.0), got 0.0",
        );
        assert_parameter_error(
            validate!(0.0 < probability <= 1.0).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in (0.0, 1.0], got 0.0",
        );

        let probability = 1.0;
        assert_parameter_error(
            validate!(0.0 <= probability < 1.0).unwrap_err(),
            "probability",
            "invalid parameter `probability`: expected in [0.0, 1.0), got 1.0",
        );
        assert_parameter_error(
            validate!(0.0 <= probability <= 0.5).unwrap_err(),
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
        let error = validate!(-1.0 < config.score <= 1.0).unwrap_err();

        assert_parameter_error(
            error,
            "score",
            "invalid parameter `score`: expected in (-1.0, 1.0], got -1.0",
        );
    }

    #[test]
    fn inclusive_range_accepts_parenthesized_expressions() {
        let lower = -1.0;
        let upper = 1.0;
        let score = 1.5;
        let error = validate!((lower) < score < (upper)).unwrap_err();

        assert_parameter_error(
            error,
            "score",
            "invalid parameter `score`: expected in (lower, upper), got 1.5",
        );
    }

    #[test]
    fn custom_checks_use_inferred_value_name() {
        let sigma = f64::NAN;
        let error = validate!(sigma, sigma.is_finite(), "finite").unwrap_err();

        assert_parameter_error(
            error,
            "sigma",
            "invalid parameter `sigma`: expected finite, got NaN",
        );
    }

    #[test]
    fn custom_checks_can_use_explicit_name() {
        let config = Some(f64::NAN);
        let error = validate!(
            "sigma",
            config,
            config.is_some_and(f64::is_finite),
            "finite"
        )
        .unwrap_err();

        assert_parameter_error(
            error,
            "sigma",
            "invalid parameter `sigma`: expected finite, got Some(NaN)",
        );
    }
}
