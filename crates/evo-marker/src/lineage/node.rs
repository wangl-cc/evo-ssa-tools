use std::{num::NonZero, rc::Rc};

use crate::Marker;

/// A node in the lineage tree.
#[derive(Debug, Default)]
pub struct LineageNode(Rc<Option<LineageNode>>);

impl LineageNode {
    /// Create a new cell with the given parent cell
    fn new(parent: LineageNode) -> Self {
        Self(Rc::new(Some(parent)))
    }

    /// Get the cell's ID.
    ///
    /// The ID is the memory address of the cell, which is unique for each cell.
    /// But it's not guaranteed to be consistent across different runs or after serialization /
    /// deserialization.
    ///
    /// This should only used to distinguish between different cells,
    /// and a permanent ID will be generated for each cell during preprocessing and serialization.
    pub(super) fn id(&self) -> NonZero<usize> {
        let addr = Rc::as_ptr(&self.0) as usize;
        // Safety: the address of Rc should not be null so its address is not zero
        unsafe { NonZero::new_unchecked(addr) }
    }

    pub(super) fn parent(&self) -> Option<&LineageNode> {
        Option::as_ref(&self.0)
    }

    /// How many references to this cell exist.
    ///
    /// For inner nodes, only children holding the Rc, so it is equal to the number of children.
    /// For leaf nodes, it is equal to 1, as cell is only hold be
    pub(super) fn ref_count(&self) -> usize {
        Rc::strong_count(&self.0)
    }
}

impl Clone for LineageNode {
    /// Clone the cell by incrementing the reference count
    ///
    /// Be careful not to clone cells accidentally, which may make the lineage inconsistent.
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl Marker for LineageNode {
    type State = ();

    fn gen_marker(&self, _: &mut Self::State) -> Self {
        Self::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_division() {
        let mut cells = vec![LineageNode::default()];

        fn divide_at(cells: &mut Vec<LineageNode>, i: usize) {
            crate::divide_at(cells, i, &mut ());
        }

        divide_at(&mut cells, 0); // cell_r divide, [1, 2]
        divide_at(&mut cells, 0); // cell_1 divide, [11, 2, 12]
        divide_at(&mut cells, 0); // cell_11 divide, [111, 2, 12, 112]
        divide_at(&mut cells, 1); // cell_2 divide, [111, 21, 12, 112, 22]
        cells.remove(0); // cell_111 die  [21, 12, 112, 22]

        assert_eq!(cells.len(), 4);

        let cell_21 = &cells[0];
        let cell_12 = &cells[1];
        let cell_112 = &cells[2];
        let cell_22 = &cells[3];

        let cell_2 = cell_21.parent().unwrap();
        let cell_0 = cell_2.parent().unwrap();
        let cell_1 = cell_12.parent().unwrap();
        let cell_11 = cell_112.parent().unwrap();

        // Check IDs, make sure they are unique and consistent
        assert_eq!(cell_0.id(), cell_1.parent().unwrap().id());
        assert_eq!(cell_2.id(), cell_22.parent().unwrap().id());
        assert_eq!(cell_1.id(), cell_11.parent().unwrap().id());
        let all_cells = [
            cell_0, cell_1, cell_2, cell_11, cell_12, cell_112, cell_21, cell_22,
        ];
        for (i, cell) in all_cells.iter().enumerate() {
            for cell_j in &all_cells[..i] {
                assert_ne!(cell.id(), cell_j.id());
            }
        }

        // Check ref count of all cells
        assert_eq!(cell_0.ref_count(), 2);
        assert_eq!(cell_1.ref_count(), 2);
        assert_eq!(cell_11.ref_count(), 1);
        assert_eq!(cell_112.ref_count(), 1);
        assert_eq!(cell_12.ref_count(), 1);
        assert_eq!(cell_2.ref_count(), 2);
        assert_eq!(cell_21.ref_count(), 1);
        assert_eq!(cell_22.ref_count(), 1);
    }
}
