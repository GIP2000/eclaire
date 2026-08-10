use crate::{
    debug,
    lexer::{LexToken, MyLexer},
    parser::{
        Parser, ParserWithState,
        grammer::{
            expression::BlockExpression,
            ident::Ident,
            types::{ConcreteType, PrimativeTypes, Type},
        },
    },
    trace,
};

#[derive(Debug, PartialEq)]
pub struct FunctionSig<'a> {
    args: Box<[(Ident<'a>, Type<'a>)]>,
    ret: Box<Type<'a>>,
}

impl<'a> Parser<'a> for FunctionSig<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        trace!("In function sig");
        _ = lexer.next_matches(LexToken::Fn)?;
        _ = lexer.next_matches(LexToken::OParen)?;

        let mut must_stop = false;
        let args: Box<_> = (|lexer: &mut L| -> anyhow::Result<_> {
            if must_stop {
                anyhow::bail!("must stop reached");
            }

            let ident = Ident::parse(lexer)?;
            _ = lexer.next_matches(LexToken::Colon)?;
            let typ = Type::parse(lexer)?;

            if let Err(_) = lexer.next_matches(LexToken::Comma) {
                must_stop = true;
            }

            Ok((ident, typ))
        })
        .parse_many(lexer)
        .map_while(Result::ok)
        .collect();

        if !must_stop && args.len() > 0 {
            return Err(super::Error::DoesNotMatch(
                "Invalid syntax for function argument",
            ));
        }

        _ = lexer.next_matches(LexToken::CParen)?;

        let ret = Box::new(if let Ok(_) = lexer.next_matches(LexToken::SkinnyArrow) {
            Type::parse(lexer)?
        } else {
            Type::from(ConcreteType::Primative(PrimativeTypes::Void))
        });

        Ok(Self { args, ret })
    }
}

#[derive(Debug)]
pub struct Function<'a> {
    sig: FunctionSig<'a>,
    block: BlockExpression<'a>,
}

impl<'a> Parser<'a> for Function<'a> {
    type Error = super::Error;

    fn from_lexer<L: MyLexer<'a>>(lexer: &mut L) -> Result<Self, Self::Error> {
        trace!("In function");
        let sig = FunctionSig::parse(lexer)?;
        let block = BlockExpression::parse(lexer)?;

        Ok(Self { sig, block })
    }
}
