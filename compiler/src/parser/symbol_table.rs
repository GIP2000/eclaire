use std::collections::HashMap;

use thiserror::Error;

use crate::{
    info,
    parser::grammer::{
        ident::Ident,
        structures::{Enum, Struct},
        Function,
    },
    trace,
};

#[derive(Debug, Error)]
pub enum SymbolTableError {
    #[error("This identifier already exists in this scope {0:?}")]
    IdentAlreadyExist(Ident),
    #[error("You can't pop the rootmost symbol table")]
    PopRoot,
}

type Result<T> = std::result::Result<T, SymbolTableError>;

#[derive(Debug)]
pub enum TypeDefInfoType {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    TypeDefPrim,
    TypeDefAlias,
}

#[derive(Debug)]
pub struct TypeInfo {
    pub size_bits: usize,
    pub type_info: TypeDefInfoType,
}

impl From<Function> for TypeInfo {
    fn from(value: Function) -> Self {
        Self {
            size_bits: 0, // TODO: make this make more sense
            type_info: TypeDefInfoType::Function(value),
        }
    }
}

#[derive(Debug, Default)]
pub struct SymbolTable {
    pub type_defs: HashMap<Ident, TypeInfo>,
    pub decls: HashMap<Ident, Ident>,
}

impl SymbolTable {
    pub fn insert_type(&mut self, key: Ident, info: TypeInfo) -> Result<()> {
        trace!("Inserting into type_defs");
        info!("Inserting {key:?} with value {info:?} into type_defs");

        self.type_defs
            .try_insert(key.clone(), info)
            .map(|_| ())
            .map_err(|_| {
                info!("ident {key:?} already exists in type_defs");
                SymbolTableError::IdentAlreadyExist(key)
            })
    }

    pub fn insert_decl(&mut self, key: Ident, type_name: Ident) -> Result<()> {
        trace!("Inserting into decl");
        info!("Inserting {key:?} with value {type_name:?} into decls");

        self.decls
            .try_insert(key.clone(), type_name)
            .map(|_| ())
            .map_err(|_| {
                info!("ident {key:?} already exists in decls");
                SymbolTableError::IdentAlreadyExist(key)
            })
    }

    pub fn get_type(&self, key: &Ident) -> Option<&TypeInfo> {
        trace!("Getting type from symbol table");
        info!("Getting type of name {key:?}");
        self.type_defs.get(key)
    }

    pub fn get_decl_name(&self, key: &Ident) -> Option<&Ident> {
        trace!("Getting a decl name from symbol table");
        info!("Getting decl of name {key:?}");
        self.decls.get(key)
    }

    pub fn get_decl(&self, key: &Ident) -> Option<&TypeInfo> {
        trace!("Getting a decl type from symbol table");
        info!("Getting decl of name {key:?}");
        self.decls.get(key).and_then(|name| self.get_type(name))
    }
}
