#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![doc = include_str!("../README.md")]

pub mod engine;
pub mod error;
pub mod model;
pub mod scheduler;

pub mod prelude {
    pub use crate::{
        engine::{RunStatus, Simulation},
        error::{Error, Result},
        model::{ChannelEditor, ChannelUpdate, EvolvingModel},
        scheduler::{
            ScheduledEvent, SsaScheduler,
            direct::{
                DenseDirectScheduler, DirectScheduler, DirectSchedulerStats, FamilyDirect,
                FamilySeparatedReactionSet, StaticDirect, StaticDirectModel, StaticReactionFamily,
                StaticReactionFamilySet,
            },
            nrm::{FamilyNrm, StaticFamilyNrm},
            update::{
                FamilyChannelUpdate, FamilyChannelUpdates, StaticChannelUpdates, StaticUpdateSink,
            },
        },
    };
}

pub use engine::{RunStatus, Simulation};
pub use error::{Error, Result};
pub use model::{ChannelEditor, ChannelUpdate, EvolvingModel};
pub use scheduler::{ScheduledEvent, SsaScheduler};
