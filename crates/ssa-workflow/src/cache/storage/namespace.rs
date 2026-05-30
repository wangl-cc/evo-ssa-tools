use crate::{
    cache::{CacheSchema, codec::ValueFormat},
    identity::ComputationPath,
};

/// Physical storage namespace derived from a computation path, value format, and cache schemas.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StorageNamespace {
    name: String,
}

impl StorageNamespace {
    /// Create a storage namespace from a computation path, value format, input schema, and output
    /// schema.
    pub fn new<I: CacheSchema, O: CacheSchema>(
        path: &ComputationPath,
        value_format: ValueFormat,
    ) -> Self {
        let name = format!(
            "{path}__{value_format}__{:08x}__{:08x}",
            I::SCHEMA_SIGNATURE,
            O::SCHEMA_SIGNATURE
        );

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
    fn storage_namespace_is_readable_and_includes_value_format_and_schemas() {
        let path = ComputationPath::root_from_str(COMPUTATION_A);
        let namespace = StorageNamespace::new::<u16, u32>(&path, FORMAT);

        assert_eq!(
            namespace.as_str(),
            format!(
                "computation-a-v1__bitcode06-v1__{:08x}__{:08x}",
                u16::SCHEMA_SIGNATURE,
                u32::SCHEMA_SIGNATURE
            )
        );
    }

    #[test]
    fn child_path_namespace_renders_leaf_first() {
        let path = ComputationPath::root_from_str("trajectory-v1").child_from_str("summary-v1");
        let namespace = StorageNamespace::new::<u16, u32>(&path, FORMAT);

        assert_eq!(
            namespace.as_str(),
            format!(
                "summary-v1_trajectory-v1__bitcode06-v1__{:08x}__{:08x}",
                u16::SCHEMA_SIGNATURE,
                u32::SCHEMA_SIGNATURE
            )
        );
    }

    #[test]
    fn namespace_includes_path_boundaries() {
        let first = ComputationPath::root_from_str("a-b").child_from_str("c");
        let second = ComputationPath::root_from_str("a").child_from_str("b-c");

        assert_ne!(
            StorageNamespace::new::<u16, u32>(&first, FORMAT),
            StorageNamespace::new::<u16, u32>(&second, FORMAT)
        );
    }

    #[test]
    fn namespace_changes_when_input_schema_changes() {
        let path = ComputationPath::root_from_str(COMPUTATION_A);

        assert_ne!(
            StorageNamespace::new::<u16, u32>(&path, FORMAT),
            StorageNamespace::new::<u32, u32>(&path, FORMAT)
        );
    }

    #[test]
    fn namespace_changes_when_output_schema_changes() {
        let path = ComputationPath::root_from_str(COMPUTATION_A);

        assert_ne!(
            StorageNamespace::new::<u16, u32>(&path, FORMAT),
            StorageNamespace::new::<u16, u64>(&path, FORMAT)
        );
    }
}
