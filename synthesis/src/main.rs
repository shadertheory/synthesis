use std::sync::atomic::{Ordering, *};
use std::mem;
use std::slice;
use bytemuck::*;

pub struct Entity {
    identity: usize,
    generation: usize,
}

static ENTITY_COUNT: AtomicUsize = AtomicUsize::new();

impl Entity {
    fn next() -> Entity {
        let identity = ENTITY_COUNT.fetch_add(1, Acquire);
        let generation = 0;

        Entity { identity, generation }
    }

    fn refresh(self) -> Entity {
        let Entity { identity, mut generation } = self;
        generation += 1;

        Entity { identity, generation }
    }
}

pub struct Type(u128);

pub struct Archetype(Vec<Type>);

pub trait Component {}

pub const CELL_SIZE: usize = 128;
pub enum Cell {
    ([u8; CELL_SIZE]),
    (Vec<u8>)
}


pub type Column = usize;

pub struct SparseSet<K: Ord, V: AnyBitPattern> {
    //Sorted list of key's and their value's column index
    indices: Vec<(K, Column)>,
    //Bit representation of all values represented contiguously
    data: Vec<u8>,

}

impl<K: Ord, V: AnyBitPattern> SparseSet<K, V> {
    pub fn new() -> SparseSet<K, V> {
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
        self.data.extend(unsafe {
            slice::from_raw_parts(&value as *const _ as *const u8, value_size)
        })
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

impl<K: Ord, V: AnyBitPattern> Default for SparseSet<K, V> {
    fn default() -> Self {
        Self {
            indices: vec![],
            data: vec![]
        }
    }
}


fn main() {
    println!("Hello, world!");
}
