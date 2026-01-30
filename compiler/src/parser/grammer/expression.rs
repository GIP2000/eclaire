use lexer::LexerIterator;

use crate::{
    debug,
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            structures::{Enum, PrimativeType, Struct},
            Function,
        },
        safe_parse_wrapper,
        symbol_table::{
            CompareTypes, SymbolTableDecl, SymbolTableError, SymbolTablePair, SymbolTableType,
            SymbolTableTypePair,
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
    TypeDefAlias(Ident),
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub size_bits: usize,
    pub type_info: TypeDefInfoType,
}

impl CompareTypes for Option<&TypeDef> {
    fn are_types_eq(&self, other: &Self, type_defs: (&SymbolTableType, usize)) -> bool {
        match (self, other) {
            (Some(a), Some(b)) => a.are_types_eq(b, type_defs),
            _ => false,
        }
    }
}

impl CompareTypes for TypeDef {
    fn are_types_eq(&self, other: &Self, type_defs: (&SymbolTableType, usize)) -> bool {
        if self.size_bits != other.size_bits {
            return false;
        }

        use TypeDefInfoType::*;

        match (&self.type_info, &other.type_info) {
            (TypeDefAlias(ident), _) | (_, TypeDefAlias(ident)) => {
                let val = type_defs.get_until_root(ident);
                val.are_types_eq(&Some(other), type_defs)
            }
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
                token_stream.parse(symbol_table).map(|x: Ident| Self {
                    size_bits: 0,
                    type_info: TypeDefInfoType::TypeDefAlias(x),
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

#[derive(Debug, Clone)]
pub enum Expression {
    BinaryOp(Box<Expression>, BinaryOperator, Box<Expression>),
    UnaryOp(UnaryOperator, Box<Expression>),

    List(Vec<Expression>),

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

#[derive(Debug, Clone, PartialEq)]
pub enum TypeResp {
    IdentRef(Ident),
    Void,
    IntLike,
    FloatLike,
}

impl CompareTypes for TypeResp {
    fn are_types_eq(&self, other: &Self, type_defs: (&SymbolTableType, usize)) -> bool {
        use TypeDefInfoType::*;
        match (self, other) {
            (TypeResp::Void, TypeResp::Void)
            | (TypeResp::IntLike, TypeResp::IntLike)
            | (TypeResp::FloatLike, TypeResp::FloatLike) => true,
            (TypeResp::IdentRef(a), TypeResp::IdentRef(b)) => {
                match (type_defs.get_until_root(a), type_defs.get_until_root(b)) {
                    (Some(a), Some(b)) => a.are_types_eq(b, type_defs),
                    _ => false,
                }
            }

            (TypeResp::IdentRef(ident), l @ TypeResp::IntLike | l @ TypeResp::FloatLike)
            | (l @ TypeResp::IntLike | l @ TypeResp::FloatLike, TypeResp::IdentRef(ident)) => {
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
        Self::IdentRef(value)
    }
}

impl Expression {
    pub fn get_type(
        &self,
        type_defs: (&SymbolTableType, usize),
        decls: &SymbolTableDecl,
    ) -> std::result::Result<TypeResp, SymbolTableError> {
        match self {
            Expression::BinaryOp(a, op, b) => {
                match op {
                    BinaryOperator::Call => {
                        let f = match a.get_type(type_defs, decls)? {
                            TypeResp::IdentRef(ident) => ident,
                            _ => return Err(SymbolTableError::TypeError),
                        };
                        let f = type_defs.get(&f).ok_or(SymbolTableError::TypeError)?;

                        if let TypeDefInfoType::Function(f) = &f.type_info {
                            match b.as_ref() {
                                Expression::List(args) => {
                                    if args.len() != f.args.len() {
                                        return Err(SymbolTableError::TypeError);
                                    }

                                    return if args
                                        .iter()
                                        .map(|arg| match arg.get_type(type_defs, decls)? {
                                            TypeResp::IdentRef(ident) => Ok(ident),
                                            _ => Err(SymbolTableError::TypeError),
                                        })
                                        .zip(f.args.iter().map(|x| &x.datatype))
                                        .all(|(a, b)| matches!((a,b), (Ok(a), b) if &a == b))
                                    {
                                        Ok(f.ret
                                            .as_ref()
                                            .cloned()
                                            .map(|x| x.into())
                                            .expect("Void functions not yet implemented"))
                                    } else {
                                        Err(SymbolTableError::TypeError)
                                    };
                                }
                                _ => return Err(SymbolTableError::TypeError),
                            };
                        }
                    }
                    BinaryOperator::Eq => {
                        if let Expression::Ident(ident) = a.as_ref() {
                            decls
                                .get(ident)
                                .ok_or(SymbolTableError::TypeError)
                                .and_then(|decl| {
                                    decl.is_mut.then_some(()).ok_or(SymbolTableError::TypeError)
                                })?
                        } else {
                            // TODO: implement mutable refrences
                            return Err(SymbolTableError::TypeError);
                        }
                    }
                    _ => (),
                }

                let type_def = a.get_type(type_defs, decls)?;

                // TODO: for now they have to be the same
                // in the future ill have implementations that I will have to check between the two
                // types to see if I can do this operation to the type
                // if type_def == b.get_type(type_defs, decls)? {
                if type_def.are_types_eq(&b.get_type(type_defs, decls)?, type_defs) {
                    Ok(type_def)
                } else {
                    Err(SymbolTableError::TypeError)
                }
            }
            Expression::UnaryOp(_, expr) => expr.get_type(type_defs, decls),
            Expression::List(_) => Err(SymbolTableError::TypeError),
            Expression::Ident(ident) => decls
                .get(ident)
                .cloned()
                .map(|x| x.ident.into())
                .ok_or(SymbolTableError::TypeError),
            Expression::Constant(ConstantExpression::IntLit(_)) => Ok(TypeResp::IntLike),
            Expression::Constant(ConstantExpression::FloatLit(_)) => Ok(TypeResp::FloatLike),
            Expression::Constant(ConstantExpression::CharLit(_)) => {
                unimplemented!("TODO: CHARLIT")
            }
            Expression::Constant(ConstantExpression::StrLit(_)) => unimplemented!("TODO: STRLIT"),
            Expression::Constant(ConstantExpression::TypeLit(_)) => {
                Err(SymbolTableError::TypeError)
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

        let ident: Result<Ident> = token_stream.parse(symbol_table);
        if let Ok(ident) = ident {
            return Ok(Expression::Ident(ident));
        }

        let constant = token_stream.parse(symbol_table)?;
        debug!("Constant = {constant:?}");

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
        let op: Option<UnaryOperator> = token_stream.parse(symbol_table).ok();

        let expr = token_stream.parse_with(symbol_table, Self::postfix_expression)?;

        Ok(match op {
            Some(x) => Expression::UnaryOp(x, Box::new(expr)),
            None => expr,
        })
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

    // conditional
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
