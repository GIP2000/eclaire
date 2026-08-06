pub struct IterPlusError<C, ErrorType>(pub C, pub Option<ErrorType>)
where
    C: Default;

impl<I, C, ErrorType> FromIterator<Result<I, ErrorType>> for IterPlusError<C, ErrorType>
where
    C: FromIterator<I> + Default,
{
    fn from_iter<T: IntoIterator<Item = Result<I, ErrorType>>>(iter: T) -> Self {
        let mut iter = iter.into_iter();
        let result = iter.by_ref().map_while(|x| x.ok()).collect();

        let following = iter.next().and_then(|x| match x {
            Ok(_) => unreachable!(
                "[FromIterator::IterPlusError]: I should have taken all the OK values already "
            ),
            Err(err) => Some(err),
        });

        IterPlusError(result, following)
    }
}
