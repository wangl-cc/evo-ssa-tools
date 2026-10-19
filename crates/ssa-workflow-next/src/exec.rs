/// Execution resources for one request: which pool runs the missing items.
///
/// `Exec` borrows the caller's pool and owns no threads. Different requests in one chain can use
/// different pools, which is how per-stage resources are expressed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exec<'p> {
    pool: Option<&'p rayon::ThreadPool>,
}

impl<'p> Exec<'p> {
    /// Run in `pool`.
    pub fn pool(pool: &'p rayon::ThreadPool) -> Self {
        Self { pool: Some(pool) }
    }

    /// Run in the current Rayon pool (the global pool outside one).
    pub fn current() -> Self {
        Self { pool: None }
    }

    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }
}
