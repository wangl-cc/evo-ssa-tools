use std::sync::atomic::{AtomicUsize, Ordering};

use canonical_input_encoding::{CanonicalBuffer, CanonicalEncode};
use parking_lot::Mutex;

use crate::{
    Error, Exec, Kernel, Keys, Path, Req, Result, Store, Upstream,
    req::{Params, Parent, Roots},
};

/// A kernel bound to a store and to its position in a chain.
///
/// Holds the computation path, the namespace prefix, the seed material, and a pool of worker
/// states that survives across blocks. Create it once per run and reuse it for every block.
pub struct Stage<'s, K: Kernel> {
    pub(crate) kernel: K,
    pub(crate) store: &'s dyn Store,
    pub(crate) path: Path,
    pub(crate) parent_path: Path,
    pub(crate) namespace: [u8; 16],
    pub(crate) seed: [u8; 32],
    pub(crate) states: Mutex<Vec<K::State>>,
    pub(crate) computed: AtomicUsize,
    pub(crate) loaded: AtomicUsize,
}

/// Counters for tests and progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageStats {
    pub computed: usize,
    pub loaded: usize,
    pub states: usize,
}

impl<'s, K: Kernel> Stage<'s, K> {
    fn new(kernel: K, store: &'s dyn Store, parent_path: Path) -> Self {
        let path = parent_path.child(K::ID);
        Self {
            kernel,
            store,
            namespace: path.namespace(),
            seed: path.seed(),
            path,
            parent_path,
            states: Mutex::new(Vec::new()),
            computed: AtomicUsize::new(0),
            loaded: AtomicUsize::new(0),
        }
    }

    /// Bind a root stage. Its inputs come from requests: `(param, repetition_index)`.
    pub fn root(kernel: K, store: &'s dyn Store) -> Self
    where
        K: Kernel<Input = ()>,
    {
        Self::new(kernel, store, Path::roots())
    }

    /// Bind a stage after `parent`. The output type of `parent` must be this kernel's input.
    pub fn after<P>(kernel: K, store: &'s dyn Store, parent: &Stage<'_, P>) -> Self
    where
        P: Kernel<Output = K::Input>,
    {
        Self::new(kernel, store, parent.path.clone())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    pub fn stats(&self) -> StageStats {
        StageStats {
            computed: self.computed.load(Ordering::Relaxed),
            loaded: self.loaded.load(Ordering::Relaxed),
            states: self.states.lock().len(),
        }
    }

    /// Request a block of root items `(param, repetition_index)`, borrowing the caller's slice.
    pub fn request_root<'a>(&'a self, items: &'a [(K::Param, u64)]) -> Req<'a, K>
    where
        K: Kernel<Input = ()>,
    {
        let roots = Roots::new(items.iter().map(|(_, rep)| *rep));
        let mut buffer = CanonicalBuffer::<K::Param>::new();
        let mut keys = Keys::with_capacity(items.len(), items.len() * (K::Param::SIZE + 8));
        for (index, (param, _)) in items.iter().enumerate() {
            keys.push_parts(buffer.encode(param), roots.keys().get(index));
        }
        Req {
            stage: self,
            parent: Parent::Owned(Box::new(roots)),
            params: Params::Pairs(items),
            keys,
            exec: Exec::current(),
        }
    }

    /// Request repetitions `reps` of one borrowed root parameter, the common
    /// "same parameter, many replicates" case. Nothing is copied per item.
    pub fn request_repeated<'a>(
        &'a self,
        param: &'a K::Param,
        reps: std::ops::Range<u64>,
    ) -> Req<'a, K>
    where
        K: Kernel<Input = ()>,
    {
        let roots = Roots::new(reps);
        let keys = derive_keys(std::slice::from_ref(param), roots.keys());
        Req {
            stage: self,
            parent: Parent::Owned(Box::new(roots)),
            params: Params::SharedRef(param),
            keys,
            exec: Exec::current(),
        }
    }

    /// Request one item per upstream item, all with the same parameter.
    pub fn request<'a>(
        &'a self,
        upstream: &'a dyn Upstream<K::Input>,
        param: K::Param,
    ) -> Result<Req<'a, K>> {
        self.check_upstream(upstream)?;
        let keys = derive_keys(std::slice::from_ref(&param), upstream.keys());
        Ok(Req {
            stage: self,
            parent: Parent::Borrowed(upstream),
            params: Params::Shared(param),
            keys,
            exec: Exec::current(),
        })
    }

    /// Fan out: every upstream item under every parameter, `params.len() × upstream.len()`
    /// items in parameter-major order. Item `p * n + u` is upstream item `u` under `params[p]`,
    /// so `values().chunks(n)` gives one chunk per parameter. Each upstream item is pulled at
    /// most once per materialization, however many parameters need it.
    pub fn request_fan<'a>(
        &'a self,
        upstream: &'a dyn Upstream<K::Input>,
        params: &'a [K::Param],
    ) -> Result<Req<'a, K>> {
        self.check_upstream(upstream)?;
        let keys = derive_keys(params, upstream.keys());
        Ok(Req {
            stage: self,
            parent: Parent::Borrowed(upstream),
            params: Params::Fan(params, upstream.keys().len()),
            keys,
            exec: Exec::current(),
        })
    }

    fn check_upstream(&self, upstream: &dyn Upstream<K::Input>) -> Result<()> {
        if upstream.path() != &self.parent_path {
            return Err(Error::WrongUpstream {
                expected: self.parent_path.to_string(),
                actual: upstream.path().to_string(),
            });
        }
        Ok(())
    }
}

/// Item `p * n + u` gets key canonical(params[p]) ++ upstream key u. Each parameter is encoded
/// once.
fn derive_keys<P: CanonicalEncode>(params: &[P], upstream: &Keys) -> Keys {
    let mut buffer = CanonicalBuffer::<P>::new();
    let items = params.len() * upstream.len();
    let mut keys = Keys::with_capacity(items, items * P::SIZE);
    for param in params {
        let head = buffer.encode(param);
        for index in 0..upstream.len() {
            keys.push_parts(head, upstream.get(index));
        }
    }
    keys
}
