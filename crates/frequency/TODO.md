# TODO

## Refactor: Reduce Template Duplication in `frequency`

### Goal

Reduce the current implementation duplication across checked vs unchecked, sequential vs parallel, and normal vs weighted frequency counting, while keeping the public API and safety boundaries easy to understand.

### Constraints

- Keep the existing public API shape unless there is a strong reason to change it.
- Keep checked and unchecked entry points explicit in the public API.
- Do not merge everything into a single large abstraction that becomes harder to read than the current code.
- Prefer private implementation sharing over public API cleverness.

### Why This Work Is Needed

The crate currently has repeated logic in several places:

- `BoundedIterator` and `UncheckedBoundedIterator`
- `BoundedParallelIterator` and `UncheckedBoundedParallelIterator`
- `freq()` and `weighted_freq()` variants
- indexed parallel constructors that apply the same thread-sized split heuristic

This duplication makes small behavior changes more expensive and increases the chance of drift between the checked and unchecked paths.

### Proposed Refactor Plan

#### 1. Keep the public API as-is

Do not collapse checked and unchecked into one public type.

Keep these explicit public entry points:

- `into_bounded_iter()`
- `into_unchecked_bounded_iter()`
- `into_bounded_par_iter()`
- `into_unchecked_bounded_par_iter()`
- `into_bounded_indexed_par_iter()`
- `into_unchecked_bounded_indexed_par_iter()`

Do the same for hash iterators, including the indexed parallel variant.

#### 2. Introduce a private bounds-access mode

Add private marker types such as:

- `Checked`
- `Unchecked`

Back them with a private trait that abstracts how a bucket is updated:

- increment a bucket
- add a weighted value to a bucket

The checked implementation should use normal indexing.

The unchecked implementation should use unchecked indexing and keep the current safety contract.

#### 3. Extract shared bounded counting helpers

Add private helper functions for the bounded counting core instead of repeating the same loops in every public impl.

Suggested split:

- `bounded_freq_seq_impl`
- `bounded_weighted_freq_seq_impl`
- `bounded_freq_par_impl`
- `bounded_weighted_freq_par_impl`

Each helper should take the private bounds-access mode as a type parameter.

This keeps the code structured around the execution model first, instead of trying to unify sequential and parallel counting into one overly abstract function.

#### 4. Extract shared indexed parallel wrapping

The indexed bounded and indexed hash parallel constructors all reuse the same thread-sized chunk heuristic.

Introduce a private helper for:

- computing the effective `min_len`
- wrapping an `IndexedParallelIterator` with `with_min_len`

Then reuse it from:

- bounded indexed checked
- bounded indexed unchecked
- hash indexed parallel

#### 5. Consider a second pass for hash internals

Hash counting currently has less duplication than bounded counting, so it is a lower-priority target.

Only refactor hash internals further if duplication remains painful after the bounded refactor.

### Non-Goals

- Do not replace the explicit checked/unchecked public API with a public `const bool` parameter.
- Do not introduce macros unless a private helper-based design still leaves obviously bad duplication.
- Do not mix safety-mode abstraction with documentation-facing type names.

### Suggested Order of Implementation

1. Extract indexed parallel wrapping helper.
2. Introduce private checked/unchecked marker types.
3. Refactor sequential bounded counting to use shared helpers.
4. Refactor parallel bounded counting to use shared helpers.
5. Re-run benchmarks to confirm there is no performance regression.
6. Clean up docs if any implementation details or examples changed.

### Validation Plan

Run at least:

- `cargo test -p frequency --features parallel`
- `cargo check -p frequency --features parallel --benches`
- `cargo bench -p frequency --bench bounded_freq --features parallel`

Compare benchmark results before and after the refactor, especially:

- bounded indexed parallel
- bounded general parallel
- hash indexed parallel
- hash general parallel

### Open Question

If the private helper design still leaves too much boilerplate after the bounded refactor, revisit whether a private `const bool` or an internal generic wrapper would simplify the code without weakening the public API.
