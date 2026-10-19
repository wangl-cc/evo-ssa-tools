use crate::{Keys, Path};

/// Materialized values of one block, in request order. Dropping it frees the values.
pub struct Block<O> {
    pub(crate) path: Path,
    pub(crate) keys: Keys,
    pub(crate) values: Vec<O>,
}

impl<O> Block<O> {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn keys(&self) -> &Keys {
        &self.keys
    }

    pub fn values(&self) -> &[O] {
        &self.values
    }

    pub fn into_values(self) -> Vec<O> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
