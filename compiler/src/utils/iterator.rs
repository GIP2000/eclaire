use crate::debug;

pub struct IterPlusError<C, ErrorType>(pub C, pub Option<ErrorType>);

impl<I, C, ErrorType> FromIterator<Result<I, ErrorType>> for IterPlusError<C, ErrorType>
where
    C: FromIterator<I>,
    ErrorType: std::fmt::Debug,
{
    fn from_iter<T: IntoIterator<Item = Result<I, ErrorType>>>(iter: T) -> Self {
        let iter = iter.into_iter();

        let mut following = None;

        let result = iter
            .map_while(|x| match x {
                x @ Ok(_) => x.ok(),
                Err(err) => {
                    debug!("{err:?}");
                    following = Some(err);
                    None
                }
            })
            .collect();

        IterPlusError(result, following)
    }
}
