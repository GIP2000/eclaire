use lexer::LexerIterator;

use crate::{
    debug,
    lexer::{LexToken, MyLexerError},
    parser::{
        grammer::{
            expression::{ConstantExpression, Expression, TypeDef},
            ident::Ident,
        },
        symbol_table::SymbolTable,
        Parse, ParseIntoWith, ParserError, ParserInto, Result,
    },
    trace,
};

#[derive(Debug, Clone, Copy)]
pub enum AssignmentType {
    Let(bool),
    Const,
}

impl Parse for AssignmentType {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<
            'a,
            crate::lexer::LexToken<'a>,
            crate::lexer::MyLexerError,
        >,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        token_stream
            .parse_with(symbol_table, |token_stream, _| {
                token_stream.next_matches(LexToken::Let)?;
                Ok(Self::Let(token_stream.next_matches(LexToken::Mut).is_ok()))
            })
            .or_else(|_| {
                token_stream
                    .next_matches(LexToken::Const)
                    .map(|_| Self::Const)
            })
            .map_err(|err| err.into())
    }
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub assignment_type: AssignmentType,
    pub ident: Ident,
    pub data_type: Option<Ident>,
    pub expr: Option<Expression>,
}

impl Parse for Assignment {
    fn from_lexer<'a>(
        token_stream: &mut impl lexer::LexerIterator<
            'a,
            crate::lexer::LexToken<'a>,
            crate::lexer::MyLexerError,
        >,
        symbol_table: &mut SymbolTable,
    ) -> Result<Self> {
        trace!("Entered Assignment");

        let assignment_type: AssignmentType = token_stream.parse(symbol_table)?;

        let (ident, data_type): (Ident, Option<Ident>) = token_stream.parse(symbol_table)?;

        let expr: Option<Expression> = token_stream
            .next_matches(LexToken::Eq)
            .ok()
            .map(|_| token_stream.parse(symbol_table))
            .transpose()?;

        use AssignmentType::*;

        match (&assignment_type, expr.as_ref()) {
            (Let(_), Some(Expression::Constant(_)))
            | (Const, None)
            | (Const, Some(Expression::List(_))) => return Err(ParserError::Other),
            (Let(_), _) => {}
            (Const, Some(Expression::Constant(ConstantExpression::TypeLit(type_data)))) => {
                symbol_table
                    .insert_type(ident.clone(), type_data.clone())
                    .map_err(|_| ParserError::Other)?;
                // TODO: ensure datatype = a type value;
            }
            (Const, Some(_)) => unimplemented!(),
        }

        token_stream.next_matches(LexToken::SemiColon)?;

        Ok(Self {
            assignment_type,
            ident,
            data_type,
            expr,
        })
    }
}
