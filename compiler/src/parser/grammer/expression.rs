use lexer::LexerIterator;

use crate::{
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            assignment::TypeRespConcrete,
            statment::Statment,
            structures::{Enum, PrimativeLike, PrimativeType, Struct},
            Function,
        },
        safe_parse_wrapper,
        symbol_table::{
            CompareTypes, STTIdxPair, SymbolTableDecl, SymbolTableError, SymbolTablePair,
            SymbolTableType, SymbolTableTypePair,
        },
        Parse, ParseIntoWith, ParserError, ParserInto, Result,
    },
    trace,
    utils::iterator::IterPlusError,
};

use super::Ident;

#[derive(Debug, Clone)]
pub enum TypeDefInfoType {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    TypeDefPrim(PrimativeType),
    TypeDefAlias(TypeRespConcrete),
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub size_bits: usize,
    pub type_info: TypeDefInfoType,
}

impl<'a> CompareTypes<'a, Option<&'a TypeDef>> for Option<&'a TypeDef> {
    fn are_types_eq(&'a self, other: &'a Self, type_defs: STTIdxPair<'_>) -> bool {
        match (self, other) {
            (Some(a), Some(b)) => a.are_types_eq(*b, type_defs),
            _ => false,
        }
    }
}

impl<'a> CompareTypes<'a, TypeDef> for TypeRespConcrete {
    fn are_types_eq(&'a self, other: &'a TypeDef, type_defs: STTIdxPair<'_>) -> bool {
        other.are_types_eq(self, type_defs)
    }
}

impl<'a> CompareTypes<'a, TypeRespConcrete> for TypeDef {
    fn are_types_eq(&'a self, other: &'a TypeRespConcrete, type_defs: STTIdxPair<'_>) -> bool {
        match (other, &self.type_info) {
            (TypeRespConcrete::IdentRef(ident), _) => type_defs
                .get_until_root(ident)
                .are_types_eq(&Some(self), type_defs),
            (a, TypeDefInfoType::TypeDefAlias(b)) => a.are_types_eq(b, type_defs),
            (TypeRespConcrete::Void, _) => false,
            (TypeRespConcrete::Pointer(_, _), _) => false,
        }
    }
}

impl<'a> CompareTypes<'a, TypeDef> for TypeDef {
    fn are_types_eq(&'a self, other: &'a Self, type_defs: STTIdxPair<'_>) -> bool {
        if self.size_bits != other.size_bits {
            return false;
        }

        use TypeDefInfoType::*;

        match (&self.type_info, &other.type_info) {
            (
                TypeDefAlias(TypeRespConcrete::Pointer(is_mut1, a)),
                TypeDefAlias(TypeRespConcrete::Pointer(is_mut2, b)),
            ) => is_mut1 == is_mut2 && a.are_types_eq(b.as_ref(), type_defs),
            (TypeDefAlias(alias), _) | (_, TypeDefAlias(alias)) => match alias {
                TypeRespConcrete::IdentRef(ident) => {
                    let val = type_defs.get_until_root(ident);
                    val.are_types_eq(&Some(other), type_defs)
                }
                x @ TypeRespConcrete::Void => x.are_types_eq(other, type_defs),
                TypeRespConcrete::Pointer(_, _) => unreachable!("I checked above"),
            },
            (Struct(s1), Struct(s2)) => s1.are_types_eq(s2, type_defs),
            (Enum(e1), Enum(e2)) => e1.are_types_eq(e2, type_defs),
            (TypeDefPrim(p1), TypeDefPrim(p2)) => p1 == p2,
            _ => false,
        }
    }
}

impl Parse for TypeDef {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<
            'a,
            crate::lexer::LexToken<'a>,
            crate::lexer::MyLexerError,
        >,
        symbol_table: &mut SymbolTableType,
    ) -> super::Result<Self> {
        trace!("Entered TypeDef");
        token_stream
            .parse(symbol_table)
            .map(|x: Function| Self {
                size_bits: 0,
                type_info: TypeDefInfoType::Function(x),
            })
            .or_else(|_| {
                token_stream.parse(symbol_table).map(|x: Struct| Self {
                    size_bits: 0,
                    type_info: TypeDefInfoType::Struct(x),
                })
            })
            .or_else(|_| {
                token_stream.parse(symbol_table).map(|x: Enum| Self {
                    size_bits: 0,
                    type_info: TypeDefInfoType::Enum(x),
                })
            })
            .or_else(|_| {
                token_stream
                    .parse(symbol_table)
                    .map(|x: PrimativeType| Self {
                        size_bits: 0,
                        type_info: TypeDefInfoType::TypeDefPrim(x),
                    })
            })
            .or_else(|_| {
                token_stream.parse_with(symbol_table, |token_stream, symbol_table| {
                    let lst: Box<_> = token_stream
                        .parse_with_many(symbol_table, |token_stream, _symbol_table| {
                            token_stream.next_matches(LexToken::Ampersand)?;
                            Ok(token_stream.next_matches(LexToken::Mut).is_ok())
                        })
                        .map_while(Result::ok)
                        .collect();

                    let type_info = lst.into_iter().fold(
                        TypeRespConcrete::IdentRef(token_stream.parse(symbol_table)?),
                        |acc, val| {
                            TypeRespConcrete::Pointer(val, Box::new(TypeResp::Concrete(acc)))
                        },
                    );

                    Ok(Self {
                        size_bits: 0,
                        type_info: TypeDefInfoType::TypeDefAlias(type_info),
                    })
                })
            })
    }
}

#[derive(Debug, Clone)]
pub enum ConstantExpression {
    IntLit(Box<str>),
    FloatLit(Box<str>),
    StrLit(Box<str>),
    CharLit(u8),
    BoolLit(bool),
    TypeLit(TypeDef),
}

impl Parse for ConstantExpression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream
            .next_matches_func(|x| {
                use LexToken::*;
                match x {
                    StrLit(x) => Some(Self::StrLit((*x).into())),
                    IntLit(x) => Some(Self::IntLit((*x).into())),
                    FloatLit(x) => Some(Self::FloatLit((*x).into())),
                    CharLit(x) => Some(Self::CharLit(*x)),
                    BoolLit(x) => Some(Self::BoolLit(*x)),
                    _ => None,
                }
            })
            .or_else(|_| {
                token_stream
                    .parse(symbol_table)
                    .map(|x: TypeDef| Self::TypeLit(x.into()))
            })
    }
}

#[derive(Debug, Clone, Default)]
pub struct BlockExpression(Vec<Statment>, usize);

impl BlockExpression {
    pub fn get_type(
        &self,
        type_defs: STTIdxPair<'_>,
        decls: &mut SymbolTableDecl,
        func_ret_type: &TypeResp,
    ) -> std::result::Result<TypeResp, SymbolTableError> {
        let Self(statments, sidx) = self;

        decls.push();

        // shadow with new context
        let type_defs = (type_defs.0, *sidx);

        let mut found = false;
        let mut first = TypeRespConcrete::Void.into();
        for s in statments.iter() {
            if let (Statment::Return(false, expr), false) = (s, found) {
                first = expr.get_type(type_defs, decls, func_ret_type)?;
                found = true;
            };
            s.type_check(type_defs, decls, func_ret_type, &first)?;
        }

        decls.pop()?;

        Ok(first)
    }
}

impl Parse for BlockExpression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream.next_matches(LexToken::OCBracket)?;

        let sidx = symbol_table.push();
        let IterPlusError(result, following) = token_stream.parse_many(symbol_table).collect();
        token_stream
            .next_matches(LexToken::CCBracket)
            .map_err(|err| following.unwrap_or(err.into()))?;

        symbol_table.pop().map_err(|_| ParserError::Other)?;

        Ok(Self(result, sidx))
    }
}

#[derive(Debug, Clone)]
pub struct FullIfExpression(Box<Expression>, BlockExpression);

#[derive(Debug, Clone)]
pub enum IfExtentionNode {
    ElseIf(FullIfExpression),
    Else(BlockExpression),
}

impl Parse for IfExtentionNode {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        token_stream.next_matches(LexToken::Else)?;

        let (expr, block) = token_stream
            .parse_with(symbol_table, |t, s| {
                Expression::base_control_flow_expression(t, s, |x| match x {
                    LexToken::If => Some(LexToken::If),
                    _ => None,
                })
            })
            .map(|(_, expr, block)| (Some(expr), block))
            .or_else(|_| token_stream.parse(symbol_table).map(|block| (None, block)))?;

        Ok(match expr {
            Some(expr) => Self::ElseIf(FullIfExpression(Box::new(expr), block)),
            None => Self::Else(block),
        })
    }
}

impl IfExtentionNode {
    pub fn get_block(&self) -> &BlockExpression {
        match self {
            IfExtentionNode::Else(b) | IfExtentionNode::ElseIf(FullIfExpression(_, b)) => b,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    BinaryOp(Box<Expression>, BinaryOperator, Box<Expression>),
    UnaryOp(UnaryOperator, Box<Expression>),

    List(Vec<Expression>),
    BlockExpression(BlockExpression),
    IfExpression(FullIfExpression, Vec<IfExtentionNode>),
    WhileExpression(Box<Expression>, BlockExpression),

    Ident(Ident),
    Constant(ConstantExpression),
}

#[inline(always)]
fn binary_op_fun_builder<'a, I, FParser, FCond>(
    parser: FParser,
    cond: FCond,
) -> impl Fn(&mut I, &mut SymbolTableType) -> Result<Expression>
where
    I: LexerIterator<'a, LexToken<'a>, MyLexerError>,
    FParser: FnMut(&mut I, &mut SymbolTableType) -> Result<Expression> + Clone,
    FCond: Fn(&BinaryOperator) -> Option<()>,
{
    #[inline(always)]
    move |token_stream: &mut I, symbol_table: &mut SymbolTableType| {
        let mut parser = parser.clone();
        let expr = safe_parse_wrapper(token_stream, symbol_table, &mut parser)?;

        Ok(token_stream
            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                let meta = token_stream
                    .clone()
                    .next()
                    .ok_or(ParserError::LexerError(
                        lexer::LexerIteratorError::NoMoreTokens,
                    ))??
                    .meta;
                let op: BinaryOperator = token_stream.parse(symbol_table)?;

                cond(&op).ok_or(ParserError::DoesNotMatch(meta.into()))?;

                let second = safe_parse_wrapper(token_stream, symbol_table, &mut parser)?;

                Ok((op, second))
            })
            .map_while(|x| x.ok())
            .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
    }
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum TypeResp {
    Concrete(TypeRespConcrete),
    IntLike,
    FloatLike,
    CharLike,
    BoolLike,
}

impl From<TypeRespConcrete> for TypeResp {
    fn from(value: TypeRespConcrete) -> Self {
        Self::Concrete(value)
    }
}

macro_rules! is_type_resp {
    ($resp: ident, $table: ident, $prim_like: pat, $type_resp_like: pat) => {
        match $resp {
            TypeResp::Concrete(TypeRespConcrete::IdentRef(ident)) => {
                let r_type = $table.get_until_root(ident);

                match r_type {
                    Some(TypeDef {
                        size_bits: _,
                        type_info:
                            TypeDefInfoType::TypeDefPrim(PrimativeType {
                                size: _,
                                like: $prim_like,
                                is_default: _,
                            }),
                    }) => true,
                    _ => false,
                }
            }
            $type_resp_like => true,
            _ => false,
        }
    };
}

impl TypeResp {
    pub fn into_concrete(
        &self,
        type_defs: &SymbolTableType,
    ) -> std::result::Result<TypeRespConcrete, SymbolTableError> {
        match self {
            TypeResp::Concrete(x) => Ok(x.clone()),
            TypeResp::IntLike => type_defs
                .default_int
                .as_ref()
                .cloned()
                .map(|x| TypeRespConcrete::IdentRef(x))
                .ok_or(SymbolTableError::TypeError("type must be known".into())),
            TypeResp::FloatLike => type_defs
                .default_float
                .as_ref()
                .cloned()
                .map(|x| TypeRespConcrete::IdentRef(x))
                .ok_or(SymbolTableError::TypeError("type must be known".into())),
            TypeResp::CharLike => type_defs
                .default_char
                .as_ref()
                .cloned()
                .map(|x| TypeRespConcrete::IdentRef(x))
                .ok_or(SymbolTableError::TypeError("type must be known".into())),
            TypeResp::BoolLike => type_defs
                .default_bool
                .as_ref()
                .cloned()
                .map(|x| TypeRespConcrete::IdentRef(x))
                .ok_or(SymbolTableError::TypeError("type must be known".into())),
        }
    }

    pub fn into_pointer(self, mutable: bool) -> Self {
        Self::Concrete(TypeRespConcrete::Pointer(mutable, Box::new(self)))
    }

    pub fn get_root_type(&self) -> &Self {
        let mut val = self;
        loop {
            match val {
                TypeResp::Concrete(TypeRespConcrete::Pointer(_, x)) => val = x.as_ref(),
                x => return x,
            }
        }
    }

    pub fn is_int(&self, type_defs: STTIdxPair<'_>) -> bool {
        is_type_resp!(
            self,
            type_defs,
            PrimativeLike::UInt | PrimativeLike::SInt,
            TypeResp::IntLike
        )
    }

    pub fn is_float(&self, type_defs: STTIdxPair<'_>) -> bool {
        is_type_resp!(self, type_defs, PrimativeLike::Float, TypeResp::FloatLike)
    }

    pub fn is_uint(&self, type_defs: STTIdxPair<'_>) -> bool {
        is_type_resp!(self, type_defs, PrimativeLike::UInt, TypeResp::IntLike)
    }

    pub fn is_bool(&self, type_defs: STTIdxPair<'_>) -> bool {
        is_type_resp!(self, type_defs, PrimativeLike::Bool, TypeResp::BoolLike)
    }
}

impl<'a> CompareTypes<'a, TypeRespConcrete> for TypeResp {
    #[inline(always)]
    fn are_types_eq(&'a self, other: &'a TypeRespConcrete, type_defs: STTIdxPair<'_>) -> bool {
        other.are_types_eq(self, type_defs)
    }
}

impl<'a> CompareTypes<'a, TypeResp> for TypeRespConcrete {
    fn are_types_eq(&'a self, other: &'a TypeResp, type_defs: STTIdxPair<'_>) -> bool {
        TypeResp::from(self.clone()).are_types_eq(other, type_defs)
    }
}

impl<'a> CompareTypes<'a, TypeResp> for TypeResp {
    fn are_types_eq(&'a self, other: &'a Self, type_defs: STTIdxPair<'_>) -> bool {
        use TypeDefInfoType::*;
        match (self, other) {
            (TypeResp::Concrete(x), TypeResp::Concrete(y)) => x.are_types_eq(y, type_defs),
            (TypeResp::IntLike, TypeResp::IntLike)
            | (TypeResp::FloatLike, TypeResp::FloatLike)
            | (TypeResp::CharLike, TypeResp::CharLike)
            | (TypeResp::BoolLike, TypeResp::BoolLike) => true,
            // (TypeResp::IdentRef(a), TypeResp::IdentRef(b)) => {
            //     match (type_defs.get_until_root(a), type_defs.get_until_root(b)) {
            //         (Some(a), Some(b)) => a.are_types_eq(b, type_defs),
            //         _ => false,
            //     }
            // }
            (
                TypeResp::Concrete(TypeRespConcrete::IdentRef(ident)),
                l @ TypeResp::IntLike
                | l @ TypeResp::FloatLike
                | l @ TypeResp::BoolLike
                | l @ TypeResp::CharLike,
            )
            | (
                l @ TypeResp::IntLike
                | l @ TypeResp::FloatLike
                | l @ TypeResp::BoolLike
                | l @ TypeResp::CharLike,
                TypeResp::Concrete(TypeRespConcrete::IdentRef(ident)),
            ) => {
                let datatype = type_defs.get_until_root(ident);

                match datatype.map(|x| &x.type_info) {
                    Some(TypeDefPrim(prim)) => TypeResp::from(prim.like) == *l,
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

impl From<Ident> for TypeResp {
    fn from(value: Ident) -> Self {
        Self::Concrete(value.into())
    }
}

impl Expression {
    pub fn get_type(
        &self,
        type_defs: STTIdxPair<'_>,
        decls: &mut SymbolTableDecl,
        func_ret_type: &TypeResp,
    ) -> std::result::Result<TypeResp, SymbolTableError> {
        match self {
            Expression::BinaryOp(a, op, b) => {
                match op {
                    BinaryOperator::Call => {
                        let f = match a.get_type(type_defs, decls, func_ret_type)? {
                            TypeResp::Concrete(TypeRespConcrete::IdentRef(ident)) => ident,
                            _ => {
                                return Err(SymbolTableError::TypeError(
                                    "Only Ident's are valid for function calls".into(),
                                ))
                            }
                        };
                        let f = type_defs.get(&f).ok_or(SymbolTableError::TypeError(
                            format!("Type `{}` not found", f).into(),
                        ))?;

                        match &f.type_info {
                            TypeDefInfoType::Function(f) => {
                                match b.as_ref() {
                                    Expression::List(args) => {
                                        if args.len() != f.args.len() {
                                            return Err(SymbolTableError::TypeError(
                                                "must supply an argument list to function".into(),
                                            ));
                                        }

                                        return if args
                                            .iter()
                                            .map(|arg| match arg.get_type(type_defs, decls, func_ret_type)? {
                                                TypeResp::Concrete(TypeRespConcrete::Void) => Err(SymbolTableError::TypeError("Invalid void argument".into())),
                                                x => Ok(x),
                                            })
                                            .zip(f.args.iter().map(|x| &x.datatype))
                                            .all(|(a, b)| matches!((a,b), (Ok(a), b) if a.are_types_eq(b, type_defs)))
                                        {
                                            Ok(f.ret
                                                .as_ref()
                                                .cloned()
                                                .map(|x| x.into())
                                                .expect("Void functions not yet implemented"))
                                        } else {
                                            Err(SymbolTableError::TypeError("Argument mismatch".into()))
                                        };
                                    }
                                    _ => {
                                        return Err(SymbolTableError::TypeError(
                                            "Only List may be proccesed".into(),
                                        ))
                                    }
                                };
                            }
                            _ => {
                                return Err(SymbolTableError::TypeError(
                                    "type must be function to call".into(),
                                ))
                            }
                        }
                    }
                    BinaryOperator::Eq => {
                        if let Expression::Ident(ident) = a.as_ref() {
                            decls
                                .get(ident)
                                .ok_or(SymbolTableError::TypeError(
                                    format!("Type `{}` not found", ident).into(),
                                ))
                                .and_then(|decl| {
                                    decl.is_mut()
                                        .then_some(())
                                        .ok_or(SymbolTableError::TypeError(
                                            "Type must be mut in order to change value".into(),
                                        ))
                                })?
                        } else {
                            // TODO: implement mutable refrences
                            return Err(SymbolTableError::TypeError(
                                "only idents can be used with an equal check".into(),
                            ));
                        }
                    }
                    _ => (),
                }

                let type_def = a.get_type(type_defs, decls, func_ret_type)?;

                // TODO: for now they mostly have to be the same
                // in the future ill have implementations that I will have to check between the two
                // types to see if I can do this operation to the type
                // if type_def == b.get_type(type_defs, decls)? {
                if type_def.are_types_eq(&b.get_type(type_defs, decls, func_ret_type)?, type_defs) {
                    if op.is_assignment() {
                        Ok(TypeResp::Concrete(TypeRespConcrete::Void))
                    } else if op.is_comparison() {
                        type_defs
                            .0
                            .default_bool
                            .as_ref()
                            .cloned()
                            .map(|bool_type| {
                                TypeResp::Concrete(TypeRespConcrete::IdentRef(bool_type))
                            })
                            .ok_or(SymbolTableError::TypeError("No boolean type set".into()))
                    } else {
                        Ok(type_def)
                    }
                } else {
                    Err(SymbolTableError::TypeError("Type mismatch".into()))
                }
            }
            Expression::UnaryOp(UnaryOperator::Not, expr) => expr
                .get_type(type_defs, decls, func_ret_type)
                .and_then(|x| {
                    x.is_bool(type_defs)
                        .then_some(x)
                        .ok_or(SymbolTableError::TypeError("Must be a bool".into()))
                }),

            Expression::UnaryOp(UnaryOperator::FromPointer, expr) => expr
                .get_type(type_defs, decls, func_ret_type)
                .and_then(|x| match x {
                    TypeResp::Concrete(TypeRespConcrete::Pointer(_, type_resp)) => Ok(*type_resp),
                    _ => Err(SymbolTableError::TypeError("type must be a pointer".into())),
                }),
            Expression::UnaryOp(UnaryOperator::IntoPointer, expr) => expr
                .get_type(type_defs, decls, func_ret_type)
                .map(|x| TypeResp::Concrete(TypeRespConcrete::Pointer(false, Box::new(x)))), // TODO: handle `& mut` vs `&`
            Expression::UnaryOp(_, expr) => expr.get_type(type_defs, decls, func_ret_type),
            Expression::List(_) => Err(SymbolTableError::TypeError(
                "Can't have a naked list".into(),
            )),
            Expression::Ident(ident) => decls
                .get(ident)
                .cloned()
                .map(|x| x.type_resp.into())
                .ok_or(SymbolTableError::TypeError("Idents must be declard".into())),
            Expression::Constant(ConstantExpression::IntLit(_)) => Ok(TypeResp::IntLike),
            Expression::Constant(ConstantExpression::FloatLit(_)) => Ok(TypeResp::FloatLike),
            Expression::Constant(ConstantExpression::CharLit(_)) => Ok(TypeResp::CharLike),
            Expression::Constant(ConstantExpression::BoolLit(_)) => Ok(TypeResp::BoolLike),
            Expression::Constant(ConstantExpression::StrLit(_)) => Ok(TypeResp::Concrete(
                TypeRespConcrete::Pointer(false, Box::new(TypeResp::CharLike)),
            )),
            Expression::Constant(ConstantExpression::TypeLit(_)) => Err(
                // TODO: comptime functions
                SymbolTableError::TypeError("Types must be removed at previous stage".into()),
            ),
            // TODO: typecheck the rest of the statments
            Expression::BlockExpression(block) => block.get_type(type_defs, decls, func_ret_type),
            Expression::IfExpression(base_if, extentions) => {
                let e_type = base_if.0.get_type(type_defs, decls, func_ret_type)?;

                e_type
                    .is_bool(type_defs)
                    .then_some(())
                    .ok_or(SymbolTableError::TypeError("must be a bool".into()))?;

                let r_type = base_if.1.get_type(type_defs, decls, func_ret_type)?;

                for ext in extentions.iter() {
                    match ext {
                        IfExtentionNode::ElseIf(FullIfExpression(expr, _)) => expr
                            .get_type(type_defs, decls, func_ret_type)
                            .and_then(|x| {
                                x.is_bool(type_defs)
                                    .then_some(())
                                    .ok_or(SymbolTableError::TypeError("Must be a bool".into()))
                            })?,
                        _ => (),
                    };
                    let ext_type = ext.get_block().get_type(type_defs, decls, func_ret_type)?;

                    ext_type
                        .are_types_eq(&r_type, type_defs)
                        .then_some(())
                        .ok_or(SymbolTableError::TypeError("".into()))?;
                }

                Ok(r_type)
            }
            Expression::WhileExpression(expr, block_expression) => {
                // TODO: handle booleans
                expr.get_type(type_defs, decls, func_ret_type)
                    .and_then(|x| {
                        x.is_bool(type_defs)
                            .then_some(())
                            .ok_or(SymbolTableError::TypeError("must be bool".into()))
                    })?;

                let r_type = block_expression.get_type(type_defs, decls, func_ret_type)?;

                r_type
                    .are_types_eq(&TypeResp::Concrete(TypeRespConcrete::Void), type_defs)
                    .then_some(TypeResp::Concrete(TypeRespConcrete::Void))
                    .ok_or(SymbolTableError::TypeError(
                        "While loop block expression types must be void for now".into(),
                    ))
            }
        }
    }

    pub fn make_binary_op(self, op: BinaryOperator, other: Self) -> Self {
        Self::BinaryOp(Box::new(self), op, Box::new(other))
    }

    fn primary_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Primary expression");

        if let Ok(_) = token_stream.next_matches(LexToken::OParen) {
            let expression: Expression = token_stream.parse(symbol_table)?;
            token_stream.next_matches(LexToken::CParen)?;
            return Ok(expression);
        }

        if let Ok(block) = token_stream.parse(symbol_table) {
            return Ok(Expression::BlockExpression(block));
        }

        if let Ok(expr) = token_stream.parse_with(symbol_table, Expression::control_flow_expression)
        {
            return Ok(expr);
        }

        let ident: Result<Ident> = token_stream.parse(symbol_table);
        if let Ok(ident) = ident {
            return Ok(Expression::Ident(ident));
        }

        let constant = token_stream.parse(symbol_table)?;

        return Ok(Expression::Constant(constant));
    }

    fn postfix_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Postfix Expression");
        let expr = token_stream.parse_with(symbol_table, Self::primary_expression)?;

        Ok(token_stream
            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                let token = token_stream.next_matches_func(|x| {
                    matches!(x, LexToken::OParen | LexToken::Dot | LexToken::OBracket)
                        .then_some(x.clone())
                })?;

                let (op, second) = match token {
                    LexToken::OParen => {
                        let IterPlusError(expr_list, following) = token_stream
                            .parse_with_many(symbol_table, |token_stream, symbol_table| {
                                let expr: Expression = token_stream.parse(symbol_table)?;
                                _ = token_stream.next_matches(LexToken::Comma);
                                Ok(expr)
                            })
                            .collect();

                        token_stream
                            .next_matches(LexToken::CParen)
                            .map_err(|err| following.unwrap_or(err.into()))?;

                        (BinaryOperator::Call, Expression::List(expr_list))
                    }
                    LexToken::Dot => {
                        let ident: Ident = token_stream.parse(symbol_table)?;
                        (BinaryOperator::Select, Expression::Ident(ident))
                    }
                    LexToken::OBracket => {
                        let expression: Expression = token_stream.parse(symbol_table)?;
                        token_stream.next_matches(LexToken::CBracket)?;
                        (BinaryOperator::ArrayIndex, expression)
                    }
                    _ => unreachable!("I checked for these tokens already"),
                };

                Ok((op, second))
            })
            .map_while(|x| x.ok())
            .fold(expr, |acc, (op, second)| acc.make_binary_op(op, second)))
    }

    fn unary_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Unary expression");
        let op: Box<[UnaryOperator]> = token_stream
            .parse_many(symbol_table)
            .map_while(Result::ok)
            .collect();

        let expr = token_stream.parse_with(symbol_table, Self::postfix_expression)?;

        Ok(op
            .into_iter()
            .fold(expr, |acc, val| Expression::UnaryOp(val, Box::new(acc))))
    }

    fn multiplicative_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering multiplicative expression");
        use BinaryOperator::*;
        binary_op_fun_builder(Self::unary_expression, |op| match op {
            Div | Mult | Mod => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn additive_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering additive expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::multiplicative_expression, |op| match op {
            Add | Sub => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn shift_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Shift Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::additive_expression, |op| match op {
            ShiftRight | ShiftLeft => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn relational_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Relational Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::shift_expression, |op| match op {
            Gt | Lt | Gte | Lte => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn equality_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Equality Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::relational_expression, |op| match op {
            BoolEq | NotEq => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn bin_and_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Binary And Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::equality_expression, |op| match op {
            BitAnd => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn xor_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Xor Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::bin_and_expression, |op| match op {
            Xor => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn bin_or_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Binary Or Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::xor_expression, |op| match op {
            BitOr => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn log_and_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Logical And Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::bin_or_expression, |op| match op {
            LogAnd => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn log_or_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Logical Or Expression");

        use BinaryOperator::*;
        binary_op_fun_builder(Self::log_and_expression, |op| match op {
            LogOr => Some(()),
            _ => return None,
        })(token_stream, symbol_table)
    }

    fn base_control_flow_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
        top: impl Fn(&LexToken<'a>) -> Option<LexToken<'a>>,
    ) -> Result<(LexToken<'a>, Expression, BlockExpression)> {
        let op = token_stream.next_matches_func(top)?;

        let expr: Expression = token_stream.parse(symbol_table)?;
        let block: BlockExpression = token_stream.parse(symbol_table)?;

        Ok((op, expr, block))
    }

    fn control_flow_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering Control Flow Expression");

        let (op, expr, block) = token_stream.parse_with(symbol_table, |t, s| {
            Self::base_control_flow_expression(t, s, |x| match x {
                y @ LexToken::If | y @ LexToken::While => Some(y.clone()),
                _ => None,
            })
        })?;

        Ok(match op {
            LexToken::If => {
                let ext: Vec<IfExtentionNode> = token_stream
                    .parse_many(symbol_table)
                    .map_while(|x| x.ok())
                    .collect();

                Self::IfExpression(FullIfExpression(Box::new(expr), block), ext)
            }
            LexToken::While => Self::WhileExpression(Box::new(expr), block),
            _ => unreachable!("I filtered this already"),
        })
    }

    fn assignment_expression<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Expression> {
        trace!("Entering assignment expression");
        token_stream
            .parse_with(symbol_table, |token_stream, symbol_table| {
                let unary = token_stream.parse_with(symbol_table, Self::unary_expression)?;

                let meta = token_stream
                    .clone()
                    .next()
                    .ok_or(ParserError::LexerError(
                        lexer::LexerIteratorError::NoMoreTokens,
                    ))??
                    .meta;
                let op: BinaryOperator = token_stream.parse(symbol_table)?;

                use BinaryOperator::*;
                match op {
                    TIMESEQ | DIVEQ | MODEQ | PLUSEQ | MINUSEQ | SHLEQ | SHREQ | ANDEQ | XOREQ
                    | OREQ | Eq => {}
                    _ => return Err(ParserError::DoesNotMatch(meta.into())),
                };

                let second = token_stream.parse_with(symbol_table, Self::log_or_expression)?;

                Ok(Expression::make_binary_op(unary, op, second))
            })
            .or_else(|_| token_stream.parse_with(symbol_table, Self::log_or_expression))
    }
}

impl Parse for Expression {
    fn from_lexer<'a>(
        token_stream: &mut impl LexerIterator<'a, LexToken<'a>, MyLexerError>,
        symbol_table: &mut SymbolTableType,
    ) -> Result<Self> {
        trace!("Entering Expression");
        Self::assignment_expression(token_stream, symbol_table)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    Pos,
    Neg,

    Not,

    FromPointer,
    IntoPointer,
}

impl<'a> TryFrom<&LexToken<'a>> for UnaryOperator {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, Self::Error> {
        match value {
            LexToken::Plus => Ok(Self::Pos),
            LexToken::Minus => Ok(Self::Neg),
            LexToken::Bang => Ok(Self::Not),

            LexToken::Ampersand => Ok(Self::IntoPointer),
            LexToken::Star => Ok(Self::FromPointer),

            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOperator {
    // multiplicative
    Mult,
    Div,
    Mod,

    // aditive
    Add,
    Sub,

    // comparison
    Gt,
    Gte,
    Lt,
    Lte,
    BoolEq,
    NotEq,
    LogAnd,
    LogOr,

    // Binary
    BitAnd,
    BitOr,
    ShiftLeft,
    ShiftRight,
    Xor,

    // Selection
    Select,
    ArrayIndex,
    Call,

    // Assignment
    TIMESEQ,
    DIVEQ,
    MODEQ,
    PLUSEQ,
    MINUSEQ,
    SHLEQ,
    SHREQ,
    ANDEQ,
    XOREQ,
    OREQ,
    Eq,
}

impl BinaryOperator {
    pub fn is_assignment(&self) -> bool {
        use BinaryOperator::*;
        matches!(
            self,
            TIMESEQ | DIVEQ | MODEQ | PLUSEQ | MINUSEQ | SHLEQ | SHREQ | ANDEQ | XOREQ | OREQ | Eq
        )
    }

    pub fn is_comparison(&self) -> bool {
        use BinaryOperator::*;
        matches!(self, Gt | Gte | Lt | Lte | BoolEq | NotEq | LogAnd | LogOr)
    }
}

impl<'a> TryFrom<&LexToken<'a>> for BinaryOperator {
    type Error = ();

    fn try_from(value: &LexToken<'a>) -> std::result::Result<Self, ()> {
        match value {
            // add sub
            LexToken::Plus => Ok(Self::Add),
            LexToken::Minus => Ok(Self::Sub),

            // mul div mod
            LexToken::Star => Ok(Self::Mult),
            LexToken::Mod => Ok(Self::Mod),
            LexToken::Div => Ok(Self::Div),

            // conditional
            LexToken::Gt => Ok(Self::Gt),
            LexToken::Gte => Ok(Self::Gte),
            LexToken::Lt => Ok(Self::Lt),
            LexToken::Lte => Ok(Self::Lte),
            LexToken::EqEq => Ok(Self::BoolEq),
            LexToken::NotEq => Ok(Self::NotEq),
            LexToken::AmpersandAmpersand => Ok(Self::LogAnd),
            LexToken::PipePipe => Ok(Self::LogOr),

            // bit
            LexToken::Pipe => Ok(Self::BitOr),
            LexToken::Ampersand => Ok(Self::BitAnd),
            LexToken::ShiftLeft => Ok(Self::ShiftLeft),
            LexToken::ShiftRight => Ok(Self::ShiftRight),
            LexToken::Carrot => Ok(Self::Xor),

            // Select
            LexToken::Dot => Ok(Self::Select),

            // Assighment
            LexToken::TIMESEQ => Ok(Self::TIMESEQ),
            LexToken::DIVEQ => Ok(Self::DIVEQ),
            LexToken::MODEQ => Ok(Self::MODEQ),
            LexToken::PLUSEQ => Ok(Self::PLUSEQ),
            LexToken::MINUSEQ => Ok(Self::MINUSEQ),
            LexToken::SHLEQ => Ok(Self::SHLEQ),
            LexToken::SHREQ => Ok(Self::SHREQ),
            LexToken::ANDEQ => Ok(Self::ANDEQ),
            LexToken::XOREQ => Ok(Self::XOREQ),
            LexToken::OREQ => Ok(Self::OREQ),
            LexToken::Eq => Ok(Self::Eq),
            _ => Err(()),
        }
    }
}
