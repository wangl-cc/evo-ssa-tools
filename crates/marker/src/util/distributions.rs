// The rand_distri::Poisson returns a float instead of an integer.
// So I need to warp it in a struct to make it return an integer.
#[derive(Clone, Copy, Debug)]
pub struct Poisson {
    exp_lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Option<Self> {
        if lambda > 0.0 {
            let exp_lambda = (-lambda).exp();
            Some(Self { exp_lambda })
        } else {
            None
        }
    }
}

impl rand::prelude::Distribution<u16> for Poisson {
    fn sample<G: rand::Rng + ?Sized>(&self, rng: &mut G) -> u16 {
        // Knuth algorithm
        let mut k = 0;
        let mut p = 1.0;
        while p > self.exp_lambda {
            k += 1;
            p *= rng.random::<f64>();
        }
        k - 1
    }
}
