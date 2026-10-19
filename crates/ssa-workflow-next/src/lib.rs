//! Prototype of the caller-driven `ssa-workflow` API.
//!
//! This crate only checks the *shape* of the design in
//! `designs/ssa-workflow-rewrite-api.md`: one generic `materialize` per stage, demand-first
//! lookups, borrowed upstream values, per-request worker counts on our own threads, and a
//! worker-state pool.
//! It uses an in-memory store and is not meant for production use.

mod block;
mod error;
mod exec;
mod kernel;
mod key;
mod path;
mod req;
mod stage;
pub mod store;
mod value;

pub use block::Block;
pub use error::{Error, Result};
pub use exec::Exec;
pub use kernel::{Cx, Kernel};
pub use key::Keys;
pub use path::Path;
pub use req::{Provided, Req, Upstream};
pub use stage::{Stage, StageStats};
pub use store::{MemoryStore, Store};
pub use value::Value;
