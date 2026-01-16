use std::{convert::Infallible, ops::FromResidual};

use crate::{lexer::MyLexerError, parser::ParserError};

type ErrorType = ParserError<MyLexerError>;

#[derive(Debug)]
pub struct IterPlusError<C: Default>(pub C, pub Option<ErrorType>);

impl<I, C: FromIterator<I> + Default> FromIterator<Result<I, ErrorType>> for IterPlusError<C> {
    fn from_iter<T: IntoIterator<Item = Result<I, ErrorType>>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let result = iter.by_ref().map_while(|x| x.ok()).collect();

        let following = iter.next().and_then(|x| match x {
            Ok(_) => unreachable!(
                "[FromIterator::IterPlusError]: I should have taken all the OK values already "
            ),
            Err(ParserError::LexerError(lexer::LexerIteratorError::NoMoreTokens)) => None,
            Err(err) => Some(err),
        });

        IterPlusError(result, following)
    }
}

impl<C: Default> FromResidual for IterPlusError<C> {
    fn from_residual(residual: <Self as std::ops::Try>::Residual) -> Self {
        Self(C::default(), Some(residual.unwrap_err()))
    }
}

impl<C: Default> std::ops::Try for IterPlusError<C> {
    type Output = C;

    type Residual = Result<Infallible, ErrorType>;

    fn from_output(output: Self::Output) -> Self {
        Self(output, None)
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self.1 {
            Some(x) => std::ops::ControlFlow::Break(Err(x)),
            None => std::ops::ControlFlow::Continue(self.0),
        }
    }
}
