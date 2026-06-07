# Validate

`validate` is a tiny crate for parameter validation in library code. It provides a diagnostic error type and a macro for common parameter checks.

## Quick Start

```rust
use validate::{ValidationError, validate};

fn configure(rate: f64, probability: f64) -> Result<(), ValidationError> {
    validate!(rate: > 0.0)?;
    validate!(probability: >= 0.0, <= 1.0)?;
    validate!(rate: rate.is_finite(); "finite")?;
    Ok(())
}
```

Failures display the parameter name, the actual value formatted with `Debug`, and a short expected condition. The parameter name is also available through `ValidationError::name()`.

```rust
use validate::{ValidationError, validate};

let steps = 0;
let error = validate!(steps: >= 1).unwrap_err();
assert_eq!(error.name(), "steps");
assert_eq!(
    error.to_string(),
    "invalid parameter `steps`: expected >= 1, got 0"
);
# let _: ValidationError = error;
```

## Supported Forms

Comparison checks put the validated identifier or field path before `:`. Field paths may use named or tuple fields. Bounds are full Rust expressions.

```rust
# use validate::{ValidationError, validate};
# let rate = 0.5;
validate!(rate: > 0.0)?;
validate!(rate: >= 0.0)?;
validate!(rate: < 1.0)?;
validate!(rate: <= 1.0)?;
# struct Config { rate: f64 }
# let config = Config { rate: 0.5 };
validate!(config.rate: > 0.0)?;
# let config = (1,);
validate!(config.0: >= 1)?;
# Ok::<(), ValidationError>(())
```

Range checks use two comma-separated clauses and render the corresponding interval. Bounds are full Rust expressions.

```rust
# use validate::{ValidationError, validate};
# let probability = 0.5;
# let min = 0.0;
# let max = 1.0;
validate!(probability: >= 0.0, <= 1.0)?;
validate!(probability: > 0.0, < 1.0)?;
validate!(probability: >= 0.0, < 1.0)?;
validate!(probability: > 0.0, <= 1.0)?;
validate!(probability: >= min, <= max)?;
# Ok::<(), ValidationError>(())
```

Custom checks use `;` to separate an arbitrary boolean expression from its static expected description.

```rust
# use validate::{ValidationError, validate};
# let sigma = 1.0_f64;
validate!(sigma: sigma.is_finite(); "finite")?;
# Ok::<(), ValidationError>(())
```
