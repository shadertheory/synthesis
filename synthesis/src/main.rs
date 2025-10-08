use bytemuck::*;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::mem;
use std::slice;
use std::sync::atomic::{Ordering::*, *};

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Entity {
    identity: usize,
    generation: usize,
}

static ENTITY_COUNT: AtomicUsize = AtomicUsize::new(0);

impl Entity {
    fn next() -> Entity {
        let identity = ENTITY_COUNT.fetch_add(1, Acquire);
        let generation = 0;

        Entity {
            identity,
            generation,
        }
    }

    fn refresh(self) -> Entity {
        let Entity {
            identity,
            mut generation,
        } = self;
        generation += 1;

        Entity {
            identity,
            generation,
        }
    }
}

pub struct Type(u128);

pub struct Archetype(Array<Type>);

pub trait Component {}

pub type Column = usize;

pub enum Array<T, const CAP: usize = 16> {
    Inline([T; CAP]),
    Spill(Vec<T>),
}

impl<T, const CAP: usize> AsRef<[T]> for Array<T, CAP> {
    fn as_ref(&self) -> &[T] {
        match self {
            Self::Inline(arr) => arr.as_ref(),
            Self::Spill(vec) => vec.as_ref(),
        }
    }
}

pub trait BitPattern: Sized {
    fn to_bytes(self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl<T: AsRef<[u8]>> BitPattern for T {
    fn to_bytes(self) -> Vec<u8> {}
    fn from_bytes(bytes: &[u8]) -> Self {}
}

pub struct MultiMap<K, V>(BTreeMap<K, Vec<V>>);

// Layer 2: Symmetric/bidirectional multimap (inverse is materialized)
pub struct SymMap<L, R> {
    forward: MultiMap<L, R>,
    reverse: MultiMap<R, L>,
}

pub struct SparseMap<K: Ord, V: BitPattern> {
    //Sorted list of key's and their value's column index
    indices: Vec<(K, Column)>,
    //Bit representation of all values represented contiguously
    data: Vec<u8>,
    value: PhantomData<[(K, V)]>,
}

impl<K: Ord, V: BitPattern> SparseMap<K, V> {
    pub fn new() -> SparseMap<K, V> {
        Default::default()
    }

    pub fn insert(&mut self, key: K, value: V) {
        let value_size = mem::size_of_val(&value);
        let total_data = self.data.len();
        let total_row_count = total_data / value_size;

        let next_column = total_row_count;

        self.indices.push((key, next_column));
        //SAFETY: value is in scope during and to the end of this function,
        //thus casting it directly to bytes is guarenteed to be valid data.
        self.data.extend(value.to_bytes())
    }

    pub fn remove(&mut self, key: &K) {
        let Ok(key_index) = self.indices.binary_search_by(|(probe, _)| probe.cmp(key)) else {
            return;
        };

        let (_, value_index) = self.indices[key_index];

        let value_size = mem::size_of::<V>();
        let column = value_index * value_size;

        self.data.splice(column..column + value_size, []);
    }
}

impl<K: Ord, V: BitPattern> Default for SparseMap<K, V> {
    fn default() -> Self {
        Self {
            indices: vec![],
            data: vec![],
            value: PhantomData,
        }
    }
}

pub type Identity = usize;

pub struct Scene {
    data: BTreeMap<Identity, SparseMap<Entity, Array<u8>>>,
    entities: SymMap<Archetype, Entity>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            data: BTreeMap::default(),
        }
    }
}

fn main() {}
