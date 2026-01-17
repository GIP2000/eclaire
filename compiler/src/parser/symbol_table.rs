use std::collections::HashMap;

use thiserror::Error;

use crate::{info, parser::grammer::ident::Ident, trace};

#[derive(Debug, Error)]
pub enum SymbolTableError {
    #[error("This identifier already exists in this scope {0:?}")]
    IdentAlreadyExist(Ident),
    #[error("You can't pop the rootmost symbol table")]
    PopRoot,
}

type Result<T> = std::result::Result<T, SymbolTableError>;

#[derive(Debug, Clone)]
pub struct TypeInfo;

#[derive(Debug, Default)]
pub struct SymbolTable {
    parent: Option<Box<SymbolTable>>,
    type_defs: HashMap<Ident, TypeInfo>,
    decls: HashMap<Ident, TypeInfo>,
}

impl SymbolTable {
    pub fn pop(&mut self) -> Result<Self> {
        trace!("Popping symbol table");
        if let Some(parent) = self.parent.take() {
            let old = std::mem::replace(self, *parent);
            info!("Sybmol Table popped succesfully");
            return Ok(old);
        }

        info!("Sybmol Table failed to pop");
        return Err(SymbolTableError::PopRoot);
    }

    pub fn new_frame(&mut self) {
        trace!("Making a new symbol table frame");

        let old = std::mem::take(self);
        self.parent = Some(Box::new(old));
    }

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

    pub fn insert_decl(&mut self, key: Ident, info: TypeInfo) -> Result<()> {
        trace!("Inserting into decl");
        info!("Inserting {key:?} with value {info:?} into decls");

        self.decls
            .try_insert(key.clone(), info)
            .map(|_| ())
            .map_err(|_| {
                info!("ident {key:?} already exists in decls");
                SymbolTableError::IdentAlreadyExist(key)
            })
    }

    pub fn get_type(&self, key: &Ident) -> Option<&TypeInfo> {
        trace!("Getting type from symbol table");
        info!("Getting type of name {key:?}");
        let mut table = Some(self);

        while let Some(table_ref) = table {
            if let Some(result) = table_ref.type_defs.get(key) {
                info!("found type of name {key:?} -> {result:?}");
                return Some(result);
            }
            table = table_ref.parent.as_ref().map(|x| x.as_ref());
        }

        info!("no type found for {key:?}");

        None
    }

    pub fn get_decl(&self, key: &Ident) -> Option<&TypeInfo> {
        trace!("Getting a decl from symbol table");
        info!("Getting decl of name {key:?}");
        let mut table = Some(self);

        while let Some(table_ref) = table {
            if let Some(result) = table_ref.decls.get(key) {
                info!("found decl of name {key:?} -> {result:?}");
                return Some(result);
            }
            table = table_ref.parent.as_ref().map(|x| x.as_ref());
        }

        info!("no decl found for {key:?}");

        None
    }
}
