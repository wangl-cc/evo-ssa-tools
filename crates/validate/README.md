# Validate

`validate` is a tiny crate for parameter validation in library code. It provides a diagnostic error type and a macro for common parameter checks.

## Quick Start

```rust
use validate::{ValidationError, validate};

fn configure(rate: f64, probability: f64) -> Result<(), ValidationError> {
    validate!(rate > 0.0)?;
    validate!(0.0 <= probability <= 1.0)?;
    validate!(rate, rate.is_finite(), "finite")?;
    Ok(())
}
```

Failures display the parameter name, the actual value formatted with `Debug`, and a short expected condition. The parameter name is also available through `ValidationError::name()`.

```rust
use validate::{ValidationError, validate};

let steps = 0;
let error = validate!(steps >= 1).unwrap_err();
assert_eq!(error.name(), "steps");
assert_eq!(
    error.to_string(),
    "invalid parameter `steps`: expected >= 1, got 0"
);
# let _: ValidationError = error;
```

## Supported Forms

Comparison checks infer the parameter name from an identifier or one field access.

```rust
# use validate::validate;
# let rate = -0.1;
let _ = validate!(rate > 0.0);
let _ = validate!(rate >= 0.0);
let _ = validate!(rate < 1.0);
let _ = validate!(rate <= 1.0);
```

Range checks use mathematical notation and render the corresponding interval.

```rust
# use validate::validate;
# let probability = 1.2;
# let min = 0.0;
# let max = 1.0;
let _ = validate!(0.0 <= probability <= 1.0);
let _ = validate!(0.0 < probability < 1.0);
let _ = validate!(0.0 <= probability < 1.0);
let _ = validate!(0.0 < probability <= 1.0);
let _ = validate!((min) <= probability <= (max));
```

Custom checks accept the value to report, a boolean condition, and a static expected description.

```rust
# use validate::validate;
# let sigma = f64::NAN;
let _ = validate!(sigma, sigma.is_finite(), "finite");
```
