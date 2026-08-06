pub mod expression;
pub mod function;
pub mod ident;
pub mod statment;
pub mod structure;
pub mod types;

pub type Error = anyhow::Error;

pub type Result<T> = core::result::Result<T, Error>;
