use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::{Block, Cx, Error, Exec, Kernel, Keys, Path, Result, Stage, Value};

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
    pub(crate) fn new(reps: &[u64]) -> Self {
        let mut keys = Keys::with_capacity(reps.len(), reps.len() * 8);
        for rep in reps {
            keys.push_parts(&rep.to_be_bytes(), &[]);
        }
        let unit = Block {
            path: Path::roots(),
            keys: keys.clone(),
            values: vec![(); reps.len()],
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

pub(crate) enum Params<P> {
    Shared(P),
    Each(Vec<P>),
}

impl<P> Params<P> {
    fn get(&self, index: usize) -> &P {
        match self {
            Self::Shared(param) => param,
            Self::Each(params) => &params[index],
        }
    }
}

/// A block of requested items for one stage. It knows its parent, so materializing it pulls
/// only the missing items from upstream.
pub struct Req<'a, K: Kernel> {
    pub(crate) stage: &'a Stage<'a, K>,
    pub(crate) parent: Parent<'a, K::Input>,
    pub(crate) params: Params<K::Param>,
    pub(crate) keys: Keys,
    pub(crate) exec: Exec<'a>,
}

impl<'a, K: Kernel> Req<'a, K> {
    /// Run this stage's missing items on `exec`.
    pub fn on(mut self, exec: Exec<'a>) -> Self {
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
        let mut raw: Vec<Option<Vec<u8>>> = vec![None; count];
        if !hit_locals.is_empty() {
            let hit_keys: Vec<&[u8]> = hit_locals
                .iter()
                .map(|&local| store_keys[local].as_slice())
                .collect();
            stage
                .store
                .visit_many_ordered(&hit_keys, &mut |position, bytes| {
                    raw[hit_locals[position]] = bytes.map(<[u8]>::to_vec);
                });
        }

        let misses: Vec<usize> = (0..count).filter(|&local| raw[local].is_none()).collect();
        let decode = |bytes: &Vec<u8>| {
            K::Output::decode(bytes).map_err(|message| Error::Decode {
                path: stage.path.to_string(),
                message,
            })
        };

        let decoded: Vec<Option<K::Output>> = self.exec.install(|| {
            raw.par_iter()
                .map(|bytes| bytes.as_ref().map(decode).transpose())
                .collect::<Result<_>>()
        })?;
        drop(raw);

        let computed: Vec<K::Output> = if misses.is_empty() {
            Vec::new()
        } else {
            // Only now ask upstream, and only for the missing items. This runs outside this
            // stage's pool, so the upstream stage finishes on its own pool before ours starts.
            let upstream_items: Vec<usize> = misses.iter().map(|&local| items[local]).collect();
            let upstream = self.parent.get().provide(&upstream_items)?;
            self.exec.install(|| {
                misses
                    .par_iter()
                    .enumerate()
                    .map_init(
                        || StateGuard::checkout(&stage.states),
                        |state, (position, &local)| {
                            let item = items[local];
                            let mut cx = Cx::new(&stage.seed, self.keys.get(item));
                            let output = stage.kernel.compute(
                                &mut cx,
                                state.get(),
                                self.params.get(item),
                                upstream.get(position),
                            )?;
                            let mut bytes = Vec::new();
                            output.encode(&mut bytes);
                            stage.store.put(&store_keys[local], bytes);
                            Ok(output)
                        },
                    )
                    .collect::<Result<_>>()
            })?
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

impl<K: Kernel> Upstream<K::Output> for Req<'_, K> {
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

/// Worker state checked out of the stage's pool for one Rayon split and returned on drop, so
/// expensive state (an assembled operator) survives across splits and blocks.
struct StateGuard<'a, S> {
    pool: &'a Mutex<Vec<S>>,
    state: Option<S>,
}

impl<'a, S: Default> StateGuard<'a, S> {
    fn checkout(pool: &'a Mutex<Vec<S>>) -> Self {
        let state = pool.lock().pop().unwrap_or_default();
        Self {
            pool,
            state: Some(state),
        }
    }

    fn get(&mut self) -> &mut S {
        self.state.as_mut().expect("state is present until drop")
    }
}

impl<S> Drop for StateGuard<'_, S> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            self.pool.lock().push(state);
        }
    }
}
