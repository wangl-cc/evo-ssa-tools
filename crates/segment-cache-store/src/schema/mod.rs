//! Persistent namespace schema shared by catalog, blocks, and segments.

mod geometry;
mod metadata;
mod value;

pub(crate) use geometry::StoreGeometry;
pub use metadata::{MetadataParseError, StoreMetadata};
pub use value::ValueLayout;
