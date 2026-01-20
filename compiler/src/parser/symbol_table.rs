use std::collections::HashMap;

use thiserror::Error;

use crate::{
    info,
    parser::grammer::{expression::TypeDef, ident::Ident},
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

#[derive(Debug, Default)]
struct SymbolTableNode {
    type_defs: HashMap<Ident, TypeDef>,
    decls: HashMap<Ident, Ident>,
}

#[derive(Debug)]
pub struct SymbolTable {
    symbol_tables: Vec<SymbolTableNode>,
    idx: usize,
    index_stack: Vec<usize>,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self {
            symbol_tables: vec![Default::default()],
            idx: 0,
            index_stack: Vec::new(),
        }
    }
}

impl SymbolTable {
    pub fn push(&mut self) -> usize {
        self.symbol_tables.push(Default::default());
        self.index_stack.push(self.idx);
        self.idx = self.symbol_tables.len() - 1;

        self.idx
    }

    pub fn pop(&mut self) -> Result<usize> {
        self.idx = self.index_stack.pop().ok_or(SymbolTableError::PopRoot)?;
        Ok(self.idx)
    }

    pub fn insert_type(&mut self, key: Ident, info: TypeDef) -> Result<()> {
        trace!("Inserting into type_defs");
        info!("Inserting {key:?} with value {info:?} into type_defs");

        self.symbol_tables[self.idx]
            .type_defs
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

        self.symbol_tables[self.idx]
            .decls
            .try_insert(key.clone(), type_name)
            .map(|_| ())
            .map_err(|_| {
                info!("ident {key:?} already exists in decls");
                SymbolTableError::IdentAlreadyExist(key)
            })
    }

    pub fn get_type(&self, key: &Ident) -> Option<&TypeDef> {
        trace!("Getting type from symbol table");
        info!("Getting type of name {key:?}");
        self.symbol_tables
            .iter()
            .rev()
            .find_map(|x| x.type_defs.get(key))
    }

    pub fn get_decl_name(&self, key: &Ident) -> Option<&Ident> {
        trace!("Getting a decl name from symbol table");
        info!("Getting decl of name {key:?}");
        self.symbol_tables
            .iter()
            .rev()
            .find_map(|x| x.decls.get(key))
    }

    pub fn get_decl(&self, key: &Ident) -> Option<&TypeDef> {
        trace!("Getting a decl type from symbol table");
        info!("Getting decl of name {key:?}");
        let name = self.get_decl_name(key)?;
        self.get_type(name)
    }
}
