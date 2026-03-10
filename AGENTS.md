# AGENTS.md

## Purpose

This file defines day-to-day execution conventions for this repository.
These conventions apply to both human contributors and coding agents.

## Repository Map

- Workspace root is managed by `Cargo.toml`.
- Rust crates live under `crates/*`.
- Each crate should document usage and behavior in its own `README.md`.

## Build, Test, and Format Commands

Use these commands as the default validation set:

```bash
cargo check
cargo test
cargo clippy --all-targets -- -D warnings
cargo +nightly llvm-cov
cargo +nightly fmt
```

Guidance:

- Prefer smallest-scope (`-p <crate>`) checks (check, clippy, test) first, then run workspace checks before final handoff.
- Use `cargo +nightly llvm-cov` for coverage runs.
- Run `cargo +nightly fmt` on touched Rust code before commit.

## Coding Guidelines (Rust 2024)

- Prefer Rust 2024 idiomatic style in all touched/new code:
  - Prefer let-chains in `if` / `while` when they improve control-flow clarity.
- Keep `unsafe` usage minimal and document invariants where `unsafe` is required.
- Do not use `unwrap` without a short explanation; either justify why it is safe here or replace it with explicit error handling.
- Keep rustdoc and README examples aligned with actual behavior and APIs.
- Do not perform broad style-only rewrites in untouched code.

## Document Guidelines

- Do not manually hard-wrap paragraphs; one paragraph in one line.

## PR & Commit Expectations

- Use Conventional Commits.
- Keep commits focused and atomic.
- PR descriptions should clearly state:
  - what changed
  - why it changed
  - which validation commands were run

## Minimal Safety Rules

- Do not run destructive git commands unless explicitly requested.
- Do not revert unrelated local changes.
- If unexpected repo changes appear, stop and ask before proceeding.

## When Uncertain

- State assumptions explicitly.
- Choose conservative defaults.
- Escalate only when ambiguity is high-impact or blocks correctness.
