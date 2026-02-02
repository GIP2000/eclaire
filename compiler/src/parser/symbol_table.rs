use std::collections::HashMap;
use std::fmt::Debug;

use thiserror::Error;

use crate::parser::grammer::assignment::TypeRespConcrete;
use crate::parser::grammer::expression::TypeResp;
use crate::{
    info,
    parser::grammer::{expression::TypeDef, ident::Ident},
    trace,
};

use super::grammer::expression::TypeDefInfoType;

#[derive(Debug, Error)]
pub enum SymbolTableError {
    #[error("Type error: {0}")]
    TypeError(Box<str>),
    #[error("This identifier already exists in this scope {0:?}")]
    IdentAlreadyExist(Ident),
    #[error("You can't pop the root of the symbol table")]
    PopRoot,
}

type Result<T> = std::result::Result<T, SymbolTableError>;

#[derive(Debug)]
pub struct SymbolTable<V> {
    symbol_tables: Vec<HashMap<Ident, V>>,
    idx: usize,
    index_stack: Vec<Vec<usize>>,
    pub default_int: Option<Ident>,
    pub default_float: Option<Ident>,
    pub default_bool: Option<Ident>,
    pub default_char: Option<Ident>,
}

pub trait SymbolTablePair<Ret> {
    fn get(&self, ident: &Ident) -> Option<&Ret>;
}

impl<V: Clone + Debug> SymbolTablePair<V> for (&SymbolTable<V>, usize) {
    fn get(&self, ident: &Ident) -> Option<&V> {
        self.0.get_as(ident, self.1)
    }
}

pub trait SymbolTableTypePair {
    fn get_until_root(&self, ident: &Ident) -> Option<&TypeDef>;
}

impl SymbolTableTypePair for (&SymbolTableType, usize) {
    fn get_until_root(&self, ident: &Ident) -> Option<&TypeDef> {
        self.0.get_as_until_root(ident, self.1)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]

pub struct DeclNode {
    pub type_resp: TypeRespConcrete,
    pub is_mut: bool,
}

impl DeclNode {
    pub fn new(type_resp: TypeRespConcrete, is_mut: bool) -> Self {
        Self { type_resp, is_mut }
    }

    pub fn is_mut(&self) -> bool {
        self.is_mut
    }
}

pub type SymbolTableType = SymbolTable<TypeDef>;
pub type SymbolTableDecl = SymbolTable<DeclNode>;

impl<V> Default for SymbolTable<V> {
    fn default() -> Self {
        Self {
            symbol_tables: vec![Default::default()],
            idx: 0,
            index_stack: vec![Vec::new()],
            default_int: None,
            default_float: None,
            default_bool: None,
            default_char: None,
        }
    }
}

impl SymbolTableDecl {
    pub fn insert_from_resp(
        &mut self,
        key: Ident,
        resp: TypeResp,
        is_mut: bool,
        type_defs: &SymbolTableType,
    ) -> Result<()> {
        self.insert(key, DeclNode::new(resp.into_concrete(type_defs)?, is_mut))
    }
}

impl SymbolTableType {
    pub fn get_as_until_root(&self, ident: &Ident, table_idx: usize) -> Option<&TypeDef> {
        let result = self.get_as(ident, table_idx)?;

        match &result.type_info {
            TypeDefInfoType::Function(_)
            | TypeDefInfoType::Struct(_)
            | TypeDefInfoType::Enum(_)
            | TypeDefInfoType::TypeDefPrim(_)
            | TypeDefInfoType::TypeDefAlias(
                TypeRespConcrete::Void | TypeRespConcrete::Pointer(_, _), // this pointer thing might break...
            ) => Some(result),
            TypeDefInfoType::TypeDefAlias(TypeRespConcrete::IdentRef(alias)) => {
                self.get_as_until_root(alias, table_idx)
            }
        }
    }

    pub fn get_until_root(&self, ident: &Ident) -> Option<&TypeDef> {
        self.get_as_until_root(ident, self.idx)
    }

    pub fn type_check(&self) -> Result<()> {
        assert_eq!(self.idx, 0, "Must be root to type check");
        let mut decls: SymbolTableDecl = SymbolTable::default();

        for idx in 0..self.symbol_tables.len() {
            self.type_check_inner(&mut decls, idx)?
        }

        Ok(())
    }

    fn type_check_inner(&self, decls: &mut SymbolTableDecl, idx: usize) -> Result<()> {
        for f in self.symbol_tables[idx]
            .iter()
            .filter_map(|(_, x)| match &x.type_info {
                TypeDefInfoType::Function(function) => Some(function),
                _ => None,
            })
        {
            let _decl_idx = decls.push();

            for arg in f.args.iter() {
                decls.insert(arg.name.clone(), DeclNode::new(arg.datatype.clone(), false))?;
            }

            let f_ret_name = f
                .ret
                .as_ref()
                .cloned()
                .map(|name| name.into())
                .unwrap_or(TypeResp::Void);

            for statment in f.statments.iter() {
                statment.type_check((self, idx), decls, &f_ret_name, &f_ret_name)?;
            }

            decls.pop()?;
        }

        Ok(())
    }
}

pub trait CompareTypes<'a, Rhs> {
    fn are_types_eq(&'a self, other: &'a Rhs, type_defs: (&SymbolTableType, usize)) -> bool;
}

impl<V> SymbolTable<V>
where
    V: Clone + Debug,
{
    pub fn push(&mut self) -> usize {
        trace!("Pushing");
        info!("symbol_table before = {:?}", self);
        self.symbol_tables.push(Default::default());
        let mut stack = self.index_stack[self.idx].clone();
        stack.push(self.idx);
        self.index_stack.push(stack);
        self.idx = self.symbol_tables.len() - 1;
        info!("symbol_table after = {:?}", self);
        self.idx
    }

    pub fn pop(&mut self) -> Result<usize> {
        trace!("Popping");
        info!("symbol_table before = {:?}", self);
        self.idx = self.index_stack[self.idx]
            .last()
            .ok_or(SymbolTableError::PopRoot)?
            .clone();
        info!("symbol_table after = {:?}", self);
        Ok(self.idx)
    }

    pub fn len(&self) -> usize {
        self.symbol_tables[self.idx].len()
    }

    pub fn insert(&mut self, key: Ident, value: V) -> Result<()> {
        trace!("Inserting into type_defs");
        info!("Inserting {key:?} with value {value:?} into type_defs");

        self.symbol_tables[self.idx]
            .try_insert(key.clone(), value)
            .map(|_| ())
            .map_err(|_| {
                info!("ident {key:?} already exists in type_defs");
                SymbolTableError::IdentAlreadyExist(key)
            })
    }

    pub fn get_as(&self, key: &Ident, table_idx: usize) -> Option<&V> {
        trace!("Getting type from symbol table");
        let result = self.index_stack[table_idx]
            .iter()
            .rev()
            .cloned()
            .chain(std::iter::once(table_idx))
            .find_map(|idx| self.symbol_tables[idx].get(key));

        info!(
            "Getting type of name {key:?} with idx {table_idx} = {:?}",
            result
        );

        result
    }

    pub fn get(&self, key: &Ident) -> Option<&V> {
        self.get_as(key, self.idx)
    }
}
