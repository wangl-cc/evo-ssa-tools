use std::sync::atomic::Ordering;

use crate::{Block, Error, Exec, Kernel, Keys, Path, Result, Stage, Store, Streams, Value};

/// Anything that can hand a stage the upstream values for some of its items: a request (which
/// materializes them on demand) or an already materialized block (which lends them).
pub trait Upstream<U>: Sync {
    fn path(&self) -> &Path;
    fn keys(&self) -> &Keys;
    /// Values for `items` (indices into `keys()`), in that order.
    fn provide(&self, items: &[usize]) -> Result<Provided<'_, U>>;
}

/// Upstream values handed to a stage: owned when they were just materialized, borrowed when the
/// caller passed an existing block.
pub enum Provided<'a, U> {
    Owned(Block<U>),
    Borrowed {
        block: &'a Block<U>,
        items: Vec<usize>,
    },
}

impl<U> Provided<'_, U> {
    fn get(&self, position: usize) -> &U {
        match self {
            Self::Owned(block) => &block.values[position],
            Self::Borrowed { block, items } => &block.values[items[position]],
        }
    }
}

impl<U: Sync> Upstream<U> for Block<U> {
    fn path(&self) -> &Path {
        &self.path
    }

    fn keys(&self) -> &Keys {
        &self.keys
    }

    fn provide(&self, items: &[usize]) -> Result<Provided<'_, U>> {
        Ok(Provided::Borrowed {
            block: self,
            items: items.to_vec(),
        })
    }
}

/// Repetition indices of root requests. Their keys are the big-endian indices.
pub(crate) struct Roots {
    keys: Keys,
    unit: Block<()>,
}

impl Roots {
    pub(crate) fn new(reps: impl Iterator<Item = u64>) -> Self {
        let mut keys = Keys::default();
        for rep in reps {
            keys.push_parts(&rep.to_be_bytes(), &[]);
        }
        let unit = Block {
            path: Path::roots(),
            keys: keys.clone(),
            values: vec![(); keys.len()],
        };
        Self { keys, unit }
    }
}

impl Upstream<()> for Roots {
    fn path(&self) -> &Path {
        &self.unit.path
    }

    fn keys(&self) -> &Keys {
        &self.keys
    }

    fn provide(&self, items: &[usize]) -> Result<Provided<'_, ()>> {
        self.unit.provide(items)
    }
}

pub(crate) enum Parent<'a, U> {
    Owned(Box<dyn Upstream<U> + 'a>),
    Borrowed(&'a dyn Upstream<U>),
}

impl<U> Parent<'_, U> {
    fn get(&self) -> &dyn Upstream<U> {
        match self {
            Self::Owned(upstream) => upstream.as_ref(),
            Self::Borrowed(upstream) => *upstream,
        }
    }
}

/// Parameters of a request. Root requests borrow the caller's items instead of copying them.
pub(crate) enum Params<'a, P> {
    Shared(P),
    SharedRef(&'a P),
    Pairs(&'a [(P, u64)]),
    /// Fan-out over `n` upstream items: item `j` uses parameter `j / n`.
    Fan(&'a [P], usize),
}

impl<P> Params<'_, P> {
    fn get(&self, index: usize) -> &P {
        match self {
            Self::Shared(param) => param,
            Self::SharedRef(param) => param,
            Self::Pairs(items) => &items[index].0,
            Self::Fan(params, n) => &params[index / n],
        }
    }
}

/// A block of requested items for one stage. It knows its parent, so materializing it pulls
/// only the missing items from upstream.
pub struct Req<'a, K: Kernel, S: Store> {
    pub(crate) stage: &'a Stage<'a, K, S>,
    pub(crate) parent: Parent<'a, K::Input>,
    pub(crate) params: Params<'a, K::Param>,
    pub(crate) keys: Keys,
    pub(crate) exec: Exec,
}

impl<'a, K: Kernel, S: Store> Req<'a, K, S> {
    /// Compute this stage's missing items with `exec` (how many workers run at once).
    pub fn on(mut self, exec: Exec) -> Self {
        self.exec = exec;
        self
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Look up, load hits, pull missing upstream values, compute misses, write them, and return
    /// every value in request order.
    pub fn materialize(&self) -> Result<Block<K::Output>> {
        let all: Vec<usize> = (0..self.keys.len()).collect();
        self.materialize_items(&all)
    }

    fn materialize_items(&self, items: &[usize]) -> Result<Block<K::Output>> {
        let stage = self.stage;
        let count = items.len();
        let store_keys: Vec<Vec<u8>> = items
            .iter()
            .map(|&item| [stage.namespace.as_slice(), self.keys.get(item)].concat())
            .collect();

        // One ordered lookup, then one ordered visit of the hits.
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_unstable_by(|a, b| store_keys[*a].cmp(&store_keys[*b]));
        let sorted: Vec<&[u8]> = order
            .iter()
            .map(|&local| store_keys[local].as_slice())
            .collect();
        let present = stage.store.contains_many_ordered(&sorted);
        let hit_locals: Vec<usize> = order
            .iter()
            .zip(&present)
            .filter(|(_, hit)| **hit)
            .map(|(local, _)| *local)
            .collect();
        // Decode inside the visit: the store hands out each value once, already decompressed,
        // and decoding is a single pass over it, so there is nothing to gain from copying the
        // bytes out to decode them in parallel later.
        let mut decoded: Vec<Option<K::Output>> =
            std::iter::repeat_with(|| None).take(count).collect();
        let mut decode_error = None;
        if !hit_locals.is_empty() {
            let hit_keys: Vec<&[u8]> = hit_locals
                .iter()
                .map(|&local| store_keys[local].as_slice())
                .collect();
            stage
                .store
                .visit_many_ordered(&hit_keys, |position, bytes| {
                    let Some(bytes) = bytes else { return };
                    if decode_error.is_some() {
                        return;
                    }
                    match K::Output::decode(bytes) {
                        Ok(value) => decoded[hit_locals[position]] = Some(value),
                        Err(message) => {
                            decode_error = Some(Error::Decode {
                                path: stage.path.to_string(),
                                message,
                            });
                        }
                    }
                });
        }
        if let Some(error) = decode_error {
            return Err(error);
        }
        let misses: Vec<usize> = (0..count)
            .filter(|&local| decoded[local].is_none())
            .collect();

        let computed: Vec<K::Output> = if misses.is_empty() {
            Vec::new()
        } else {
            // Only now ask upstream, and only for the missing items. This runs before this
            // stage's workers start, so the upstream stage finishes with its own workers first.
            // Item `j` comes from upstream item `j % n` (fan-out is parameter-major; without
            // fan-out this is the identity). Several misses may share one upstream item under
            // fan-out; ask for each once. One dynamic call for the length, not one per item.
            let parent = self.parent.get();
            let n = parent.keys().len();
            let upstream_of = |item: usize| item % n;
            let mut wanted: Vec<usize> = misses
                .iter()
                .map(|&local| upstream_of(items[local]))
                .collect();
            wanted.sort_unstable();
            wanted.dedup();
            let positions: Vec<usize> = misses
                .iter()
                .map(|&local| {
                    wanted
                        .binary_search(&upstream_of(items[local]))
                        .expect("wanted holds every needed upstream item")
                })
                .collect();
            let upstream = parent.provide(&wanted)?;
            self.exec.map(
                misses.len(),
                || stage.states.lock().pop().unwrap_or_default(),
                |state| stage.states.lock().push(state),
                |state, miss| {
                    let local = misses[miss];
                    let item = items[local];
                    let mut streams = Streams::new(&stage.seed, self.keys.get(item));
                    let output = stage.kernel.compute(
                        &mut streams,
                        state,
                        self.params.get(item),
                        upstream.get(positions[miss]),
                    )?;
                    let mut bytes = Vec::new();
                    output.encode(&mut bytes);
                    stage.store.put(&store_keys[local], bytes);
                    Ok(output)
                },
            )?
            // `upstream` is dropped here: an owned upstream block lives only while this stage
            // computes.
        };

        stage
            .loaded
            .fetch_add(count - misses.len(), Ordering::Relaxed);
        stage.computed.fetch_add(misses.len(), Ordering::Relaxed);

        let mut computed = computed.into_iter();
        let values = decoded
            .into_iter()
            .map(|value| {
                value.unwrap_or_else(|| computed.next().expect("one computed value per miss"))
            })
            .collect();
        Ok(Block {
            path: stage.path.clone(),
            keys: self.keys.select(items),
            values,
        })
    }
}

impl<K: Kernel, S: Store> Upstream<K::Output> for Req<'_, K, S> {
    fn path(&self) -> &Path {
        &self.stage.path
    }

    fn keys(&self) -> &Keys {
        &self.keys
    }

    fn provide(&self, items: &[usize]) -> Result<Provided<'_, K::Output>> {
        self.materialize_items(items).map(Provided::Owned)
    }
}
