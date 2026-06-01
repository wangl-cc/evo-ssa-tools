pub(crate) const KEY_LEN: usize = 128;

#[derive(Clone, Copy)]
pub(crate) enum ValueProfile {
    Small,
    SmallFixed,
    Medium,
    Large,
}

impl ValueProfile {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::SmallFixed => "small_fixed",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    pub(crate) fn base_len(self) -> usize {
        match self {
            Self::Small => 64,
            Self::SmallFixed => 64,
            Self::Medium => 1_024,
            Self::Large => 16 * 1_024,
        }
    }

    pub(crate) fn jitter(self) -> usize {
        match self {
            Self::Small => 8,
            Self::SmallFixed => 0,
            Self::Medium => 128,
            Self::Large => 2 * 1_024,
        }
    }

    pub(crate) fn uses_large_value_tuning(self) -> bool {
        matches!(self, Self::Large)
    }

    pub(crate) fn fixed_value_len(self) -> Option<usize> {
        match self {
            Self::SmallFixed => Some(self.base_len()),
            Self::Small | Self::Medium | Self::Large => None,
        }
    }
}

pub(crate) const PROFILES: &[ValueProfile] = &[
    ValueProfile::Small,
    ValueProfile::SmallFixed,
    ValueProfile::Medium,
    ValueProfile::Large,
];
