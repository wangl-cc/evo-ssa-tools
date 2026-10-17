#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

/// Simulation engines and run termination state.
pub mod engine;
/// Errors returned by models, schedulers, and simulations.
pub mod error;
/// Dynamic-channel model contracts and update types.
pub mod model;
/// Independent random streams for scheduler and model draws.
pub mod random;
/// Exact SSA scheduler implementations and static-family contracts.
pub mod scheduler;

/// Commonly used model, scheduler, simulation, and macro exports.
pub mod prelude {
    pub use crate::{
        engine::{RunStatus, Simulation},
        error::{Error, Result},
        family_list,
        model::{ChannelEditor, ChannelUpdate, EvolvingModel},
        random::SsaRngs,
        reaction_families,
        scheduler::{
            ScheduledEvent, SsaScheduler,
            direct::{DirectScheduler, DirectSchedulerStats, FamilyDirect},
            family::{FamilyId, StaticFamilyModel, StaticReactionFamily},
            nrm::{FamilyNrm, StaticFamilyNrm},
            update::ChannelRecomputeSink,
        },
    };
}

pub use engine::{RunStatus, Simulation};
pub use error::{Error, Result};
pub use model::{ChannelEditor, ChannelUpdate, EvolvingModel};
pub use random::SsaRngs;
pub use scheduler::{ScheduledEvent, SsaScheduler};
