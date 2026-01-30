use std::{
    fmt::Display,
    ops::{BitAnd, BitOr},
    sync::OnceLock,
};

#[rustfmt::skip]
mod log_levels {
    pub(crate) const ALL  : u8 = 0b00000001;
    pub(crate) const TRACE: u8 = 0b00000010;
    pub(crate) const DEBUG: u8 = 0b00000100;
    pub(crate) const INFO : u8 = 0b00001000;
    pub(crate) const WARN : u8 = 0b00010000;
    pub(crate) const ERROR: u8 = 0b00100000;
    pub(crate) const FATAL: u8 = 0b01000000;
}

use log_levels::*;

static LOG_LEVEL: OnceLock<Level> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level(u8);

impl Level {
    pub fn set_log_level(value: u8) {
        LOG_LEVEL
            .set(Level(value))
            .expect("COULD NOT SET LOG LEVEL");
    }

    pub const fn none() -> Self {
        Self(0)
    }
    pub const fn all() -> Self {
        Self(ALL)
    }
    pub const fn trace() -> Self {
        Self(TRACE)
    }
    pub const fn debug() -> Self {
        Self(DEBUG)
    }
    pub const fn info() -> Self {
        Self(INFO)
    }
    pub const fn warn() -> Self {
        Self(WARN)
    }
    pub const fn error() -> Self {
        Self(ERROR)
    }
    pub const fn fatal() -> Self {
        Self(FATAL)
    }
}

impl Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.count_ones() != 1 {
            write!(f, "{:?}", self)
        } else if *self & Self::all() == Self::all() {
            write!(f, "[ALL]")
        } else if *self & Self::trace() == Self::trace() {
            write!(f, "[TRACE]")
        } else if *self & Self::debug() == Self::debug() {
            write!(f, "[DEBUG]")
        } else if *self & Self::info() == Self::info() {
            write!(f, "[INFO]")
        } else if *self & Self::warn() == Self::warn() {
            write!(f, "[WARN]")
        } else if *self & Self::error() == Self::error() {
            write!(f, "[ERROR]")
        } else if *self & Self::fatal() == Self::fatal() {
            write!(f, "[FATAL]")
        } else {
            write!(f, "{:?}", self)
        }
    }
}

impl From<Level> for u8 {
    fn from(value: Level) -> Self {
        value.0
    }
}

impl BitOr<Level> for Level {
    type Output = u8;

    fn bitor(self, rhs: Level) -> Self::Output {
        self.0 | rhs.0
    }
}

impl BitOr<u8> for Level {
    type Output = u8;

    fn bitor(self, rhs: u8) -> Self::Output {
        self.0 | rhs
    }
}

impl BitOr<Level> for u8 {
    type Output = u8;

    fn bitor(self, rhs: Level) -> Self::Output {
        self | rhs.0
    }
}

impl BitAnd<Level> for Level {
    type Output = u8;

    fn bitand(self, rhs: Level) -> Self::Output {
        self.0 & rhs.0
    }
}

impl PartialEq<Level> for u8 {
    fn eq(&self, other: &Level) -> bool {
        *self == other.0
    }
}

#[macro_export]
macro_rules! log {
    ($level: expr, $($arg:tt)*) => {
        $crate::utils::logger::log($level, file!(), line!(), format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! all {
        ($($arg:tt)*) => {
         $crate::log!($crate::utils::logger::Level::all(), $($arg)*)
        };
    }

#[macro_export]
macro_rules! trace{
        ($($arg:tt)*) => {
            $crate::log!($crate::utils::logger::Level::trace(), $($arg)*)
        };
    }

#[macro_export]
macro_rules! debug{
        ($($arg:tt)*) => {
            $crate::log!($crate::utils::logger::Level::debug(), $($arg)*)
        };
    }

#[macro_export]
macro_rules! info{
        ($($arg:tt)*) => {
            $crate::log!($crate::utils::logger::Level::info(), $($arg)*)
        };
    }

#[macro_export]
macro_rules! warn{
        ($($arg:tt)*) => {
            $crate::log!($crate::utils::logger::Level::warn(), $($arg)*)
        };
    }

#[macro_export]
macro_rules! error{
        ($($arg:tt)*) => {
            $crate::log!($crate::utils::logger::Level::error(), $($arg)*)
        };
    }

#[macro_export]
macro_rules! fatal{
        ($($arg:tt)*) => {
            $crate::log!($crate::utils::logger::Level::fatal(), $($arg)*)
        };
    }

#[inline]
pub fn log(level: Level, file: &str, line: u32, args: std::fmt::Arguments) {
    #[cfg(not(test))]
    {
        let allowed_log = LOG_LEVEL.get().expect("ERROR UNITILIZED LOG LEVEL");

        if (*allowed_log & level) <= 0 {
            return;
        };
        println!("{} [{}:{}]: {}", level, file, line, args);
    }
    #[cfg(test)]
    eprintln!("{} [{}:{}]: {}", level, file, line, args);
}
