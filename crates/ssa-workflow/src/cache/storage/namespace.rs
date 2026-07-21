use canonical_input_encoding::CANONICAL_INPUT_ENCODING_VERSION;

use crate::{cache::codec::ValueFormat, identity::ComputationPath};

/// Physical storage namespace derived from a computation path, canonical input encoding, and
/// value format.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageNamespace {
    name: String,
}

impl StorageNamespace {
    /// Create a storage namespace from a computation path and value format.
    ///
    /// The namespace includes the canonical input encoding version so that persistent caches
    /// written under an older encoding are never read through the new namespace.
    pub fn new(path: &ComputationPath, value_format: ValueFormat) -> Self {
        let name = format!("{path}__{CANONICAL_INPUT_ENCODING_VERSION}__{value_format}");

        Self { name }
    }

    /// Return the storage namespace name.
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::identity::ComputationPath;

    const COMPUTATION_A: &str = "computation-a-v1";
    const FORMAT: ValueFormat = ValueFormat::new("bitcode06-v1");

    #[test]
    fn storage_namespace_includes_key_format_and_value_format() {
        let path = ComputationPath::root_from_str(COMPUTATION_A);
        let namespace = StorageNamespace::new(&path, FORMAT);

        assert_eq!(
            namespace.as_str(),
            "computation-a-v1__keyfmt-v2__bitcode06-v1"
        );
    }

    #[test]
    fn child_path_namespace_renders_leaf_first() {
        let path = ComputationPath::root_from_str("trajectory-v1").child_from_str("summary-v1");
        let namespace = StorageNamespace::new(&path, FORMAT);

        assert_eq!(
            namespace.as_str(),
            "summary-v1_trajectory-v1__keyfmt-v2__bitcode06-v1"
        );
    }

    #[test]
    fn namespace_includes_path_boundaries() {
        let first = ComputationPath::root_from_str("a-b").child_from_str("c");
        let second = ComputationPath::root_from_str("a").child_from_str("b-c");

        assert_ne!(
            StorageNamespace::new(&first, FORMAT),
            StorageNamespace::new(&second, FORMAT)
        );
    }
}
