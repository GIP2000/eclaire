use std::fmt::Debug;

#[derive(Default, Eq)]
pub(crate) struct VecSet<T>
where
    T: Eq,
{
    arr: Vec<T>,
}

impl<E: Eq> Extend<E> for VecSet<E> {
    fn extend<T: IntoIterator<Item = E>>(&mut self, iter: T) {
        for val in iter {
            self.insert(val);
        }
    }
}

impl<const S: usize, T: Eq> From<[T; S]> for VecSet<T> {
    fn from(value: [T; S]) -> Self {
        Self { arr: value.into() }
    }
}

impl<T: Eq> PartialEq for VecSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.arr.len() == other.arr.len() && self.iter().all(|x| other.contains(x))
    }
}

impl<T: Debug + Eq> Debug for VecSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.arr.fmt(f)
    }
}

impl<T: Clone + Eq> Clone for VecSet<T> {
    fn clone(&self) -> Self {
        Self {
            arr: self.arr.clone(),
        }
    }
}

impl<T> VecSet<T>
where
    T: Eq,
{
    pub fn new() -> Self {
        Self { arr: vec![] }
    }

    pub fn len(&self) -> usize {
        self.arr.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> {
        self.arr.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.arr.into_iter()
    }

    pub fn insert(&mut self, val: T) {
        if self.iter().find(|&x| x == &val).is_none() {
            self.arr.push(val);
        }
    }

    pub fn contains(&self, val: &T) -> bool {
        self.iter().find(|&x| x == val).is_some()
    }

    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.arr
            .iter()
            .chain(other.arr.iter().filter(|x| !self.contains(x)))
    }
}

impl<E: Eq> FromIterator<E> for VecSet<E> {
    fn from_iter<T: IntoIterator<Item = E>>(iter: T) -> Self {
        Self {
            arr: iter.into_iter().collect(),
        }
    }
}

pub(crate) struct VecMap<K, V>
where
    K: Eq,
{
    arr: Vec<(K, V)>,
}

impl<K: Eq + Debug, V: Debug> Debug for VecMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.arr.fmt(f)
    }
}

impl<K, V> VecMap<K, V>
where
    K: Eq,
{
    pub fn new() -> Self {
        Self { arr: vec![] }
    }

    pub fn len(&self) -> usize {
        self.arr.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a (K, V)> {
        self.arr.iter()
    }

    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.arr.into_iter()
    }

    pub fn values<'a>(&'a self) -> impl Iterator<Item = &'a V> {
        self.iter().map(|(_, v)| v)
    }

    pub fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K> {
        self.iter().map(|(k, _)| k)
    }

    pub fn insert(&mut self, key: K, value: V) {
        let val = self.arr.iter_mut().find(|(k, _)| k == &key);

        match val {
            Some((_, v)) => *v = value,
            None => self.arr.push((key, value)),
        }
    }

    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.arr.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.arr.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}
