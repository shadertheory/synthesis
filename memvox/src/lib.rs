#![feature(trait_alias)]
use arrayvec::ArrayVec as Array;
use math::*;
use std::cell::*;
use std::{any::TypeId, array, collections::HashMap, mem, ops::*, process::Output};

use crate::palette::Palette;

pub trait Bit =
    Sized + BitOr<Self, Output = Self> + BitOrAssign<Self> + BitAnd<Self, Output = Self>;

pub trait Sh<T> = Shl<T, Output = Self> + Shr<T, Output = Self>;

pub trait Basic = Sized + Add<Self, Output = Self> + Sub<Self, Output = Self>;

pub trait Fundamental = Not<Output = Self> + Copy + Zero + One + Basic + Convert<usize>;

pub trait Identity {
    fn identity(self) -> Self;
}

impl<T> Identity for T {
    fn identity(self) -> Self {
        self
    }
}

pub mod palette {
    use crate::*;
    use arrayvec::ArrayVec as Array;
    use std::fmt::Debug;
    use std::iter;
    use std::mem::{self, MaybeUninit};
    use std::slice;

    // Hardcoding u32 to prevent trait logic errors
    type R = u32;
    const BITS: usize = 32;

    #[derive(Debug, Clone)]
    pub struct Palette<Compress, const CAP: usize = 16> {
        palette: Array<Compress, CAP>,
        len: usize,
        // We store absolute number of items, not raw integers,
        // to strictly control memory layout
        compressed: Vec<u32>,
    }

    impl<Compress: std::fmt::Debug + Ord + Clone> Index<usize> for Palette<Compress> {
        type Output = Compress;
        fn index(&self, index: usize) -> &Self::Output {
            self.palettize(&self.get(index)).unwrap()
        }
    }

    impl<Compress: std::fmt::Debug + Clone + Ord + PartialEq, const CAP: usize> Palette<Compress, CAP> {
        pub fn new(repr: Compress, count: usize) -> Self {
            // Create a palette with just one item initially
            let mut pal = Array::new();
            pal.push(repr);

            // If only 1 item, we need 0 bits, so compressed is empty
            Self {
                palette: pal,
                len: count,
                compressed: vec![],
            }
        }

        fn calculate_bits(palette_len: usize) -> usize {
            if palette_len <= 1 {
                return 0;
            }
            (palette_len as f32).log2().ceil() as usize
        }

        fn write(buf: &mut [u32], index_placement: usize, val: u32, bits: usize) {
            if bits == 0 {
                return;
            }

            let global_bit_pos = index_placement * bits;
            let index_repr = global_bit_pos / BITS;
            let shift = global_bit_pos % BITS;

            // Mask representing the value's width (e.g., 111)
            let val_mask = (1u32 << bits).wrapping_sub(1);

            // 1. Write to the main integer
            let current = buf[index_repr];
            // Create a mask to CLEAR the target spot (e.g., 111000111)
            let clear_mask = !(val_mask << shift);
            // Shift our new value into place
            let new_val = (val & val_mask) << shift;

            // Combine: (Old & Clear) | New
            buf[index_repr] = (current & clear_mask) | new_val;

            // 2. Handle Overflow (if value spans two integers)
            if shift + bits > BITS {
                let bits_first = BITS - shift;
                let bits_second = bits - bits_first;

                let next_idx = index_repr + 1;
                if next_idx < buf.len() {
                    let current_next = buf[next_idx];

                    // Mask for the bits going into the next int
                    let val_mask_second = (1u32 << bits_second).wrapping_sub(1);
                    let clear_mask_next = !val_mask_second; // Clear bottom N bits

                    // Shift the original value DOWN to get the top bits
                    let new_val_next = val >> bits_first;

                    buf[next_idx] = (current_next & clear_mask_next) | new_val_next;
                }
            }
        }

        fn read(buf: &[u32], index_placement: usize, bits: usize) -> u32 {
            if bits == 0 {
                return 0;
            }

            let global_bit_pos = index_placement * bits;
            let index_repr = global_bit_pos / BITS;
            let shift = global_bit_pos % BITS;

            let val_mask = (1u32 << bits).wrapping_sub(1);

            // Read low part
            let val_low = buf[index_repr] >> shift;

            if shift + bits <= BITS {
                val_low & val_mask
            } else {
                // Read high part (spillover)
                let bits_first = BITS - shift;
                let val_high = buf[index_repr + 1]; // This is bit 0 of next int

                // We take the low bits of the next int, and shift them UP
                // to become the high bits of our result
                let high_part = val_high << bits_first;

                (val_low | high_part) & val_mask
            }
        }

        pub fn compress(blocks: &[Compress]) -> Self {
            let mut palette_vec = blocks.to_vec();
            palette_vec.sort();
            palette_vec.dedup();

            let palette = Array::<Compress, CAP>::from_iter(palette_vec.iter().cloned());
            let len = blocks.len();
            let bits = Self::calculate_bits(palette.len());

            let mut compressed = vec![];

            if bits > 0 {
                let total_bits = len * bits;
                let u_count = (total_bits + BITS - 1) / BITS;
                // Initialize strictly with 0
                compressed = vec![0u32; u_count];

                for (place, block) in blocks.iter().enumerate() {
                    // Find index in palette
                    let mut idx = 0;
                    for (i, p) in palette.iter().enumerate() {
                        if p == block {
                            idx = i;
                            break;
                        }
                    }
                    Self::write(&mut compressed, place, idx as u32, bits);
                }
            }

            Self {
                palette,
                len,
                compressed,
            }
        }

        pub fn decompress(&self) -> Vec<Compress> {
            let bits = Self::calculate_bits(self.palette.len());
            let mut ret = Vec::with_capacity(self.len);

            // Optimization for single-block palette
            if bits == 0 {
                let block = self.palette[0].clone();
                ret.resize(self.len, block);
                return ret;
            }

            for place in 0..self.len {
                let idx = Self::read(&self.compressed, place, bits);
                ret.push(self.palette[idx as usize].clone());
            }
            ret
        }

        pub fn get(&self, index: usize) -> Compress {
            let bits = Self::calculate_bits(self.palette.len());
            // Fast path for uniform palette
            if bits == 0 {
                return self.palette[0].clone();
            }
            let idx = Self::read(&self.compressed, index, bits);
            self.palette[idx as usize].clone()
        }

        pub fn set(&mut self, index: usize, block: Compress) {
            // 1. Is block already in palette?
            let palette_idx = self.palette.iter().position(|p| p == &block);

            match palette_idx {
                Some(pid) => {
                    // Block exists in palette, just write the index
                    let bits = Self::calculate_bits(self.palette.len());
                    Self::write(&mut self.compressed, index, pid as u32, bits);
                }
                None => {
                    // Block NOT in palette. We must decompress, add, and re-compress.
                    let mut full_data = self.decompress();
                    full_data[index] = block;
                    *self = Self::compress(&full_data);
                }
            }
        }

        pub fn contains(&self, block: &Compress) -> bool {
            self.palette.contains(block)
        }

        pub fn palettize(&self, block: &Compress) -> Option<&Compress> {
            self.palette.iter().find(|p| *p == block)
        }
    }
}
pub trait Volume<const DIM: usize, Subtype> {
    fn get(&self, position: Vector<DIM, usize>) -> &Subtype;
    fn set(&self, position: Vector<DIM, usize>, block: Subtype);
    fn capacity(&self) -> Vector<DIM, usize>;
    fn origin(&self) -> Vector<DIM, i64>;
}

pub trait Voxel {
    fn identifier(&self) -> u64 {
        0
    }
}

pub struct MemVol<T> {
    capacity: Vector<3, usize>,
    palette: UnsafeCell<Palette<T>>,
}

impl<T: std::fmt::Debug> MemVol<T> {
    pub fn with_capacity(repr: T, capacity: Vector<3, usize>) -> Self
    where
        T: Ord + Clone,
    {
        Self {
            capacity,
            palette: Palette::new(repr, capacity.iter().fold(1, Mul::mul)).into(),
        }
    }
}

impl<T: std::fmt::Debug> Index<(usize, usize, usize)> for MemVol<T>
where
    T: Ord + Clone,
{
    type Output = T;
    fn index(&self, idx: (usize, usize, usize)) -> &<Self as Index<(usize, usize, usize)>>::Output {
        let (x, y, z) = idx;
        let palette = unsafe { self.palette.get().as_mut().unwrap() };

        palette
            .palettize(&palette.get(Vector([x, y, z]).to_one_dim(self.capacity)))
            .unwrap()
    }
}

pub struct Erasure {
    ty: TypeId,
    data: Array<u8, 8>,
}

pub struct Entry {
    pub normalized: String,
    pub attributes: HashMap<usize, Erasure>,
}

pub struct Attribute {
    range: Range<usize>,
    default: usize,
    ty: TypeId,
    name: String,
    description: usize,
}

pub struct MemVox {
    attributes: Vec<Attribute>,
    entries: Vec<Entry>,
}

impl<T: Ord + Clone + std::fmt::Debug> Volume<3, T> for MemVol<T> {
    fn capacity(&self) -> Vector<3, usize> {
        self.capacity
    }
    fn origin(&self) -> Vector<3, i64> {
        Vector([0, 0, 0])
    }
    fn set(&self, position: Vector<3, usize>, vox: T) {
        let palette = unsafe { self.palette.get().as_mut().unwrap() };
        let index = position.to_one_dim(self.capacity);
        if palette.contains(&vox) {
            palette.set(index, vox);
        } else {
            let mut data = palette.decompress();
            data[index] = vox;
            *palette = Palette::compress(&data);
        }
    }
    fn get(&self, position: Vector<3, usize>) -> &T {
        let palette = unsafe { self.palette.get().as_mut().unwrap() };
        dbg!(position, self.capacity);
        &palette[position.to_one_dim(self.capacity)]
    }
}
#[derive(Debug, Clone)]
pub struct BoundingBox<T> {
    pub min: Vector<3, usize>,
    pub size: Vector<3, usize>,
    pub block: T,
}
impl<T: PartialEq + Clone + std::fmt::Debug> BoundingBox<T> {
    pub fn derive(vol: &dyn Volume<3, T>, visible: impl Fn(&T) -> bool) -> Vec<BoundingBox<T>> {
        let cap = vol.capacity();
        let max_len = cap.iter().fold(1, |acc, &x| acc * x);
        let mut visited = vec![false; max_len];
        let mut ret = vec![];

        println!("--- STARTING MESH GENERATION ---");

        for i in 0..max_len {
            if visited[i] {
                continue;
            }

            let pos = Vector::from_one_dim(i, cap);
            let start_block = vol.get(pos);

            if !(visible)(start_block) {
                visited[i] = true;
                continue;
            }

            println!("Found Start: {:?} @ {:?}", start_block, pos);

            // 1. Expand X
            let mut size_z = 0;
            while pos[0] + size_z < cap[0] {
                let check_pos = pos + Vector([0, 0, size_z]);
                let check_block = vol.get(check_pos);
                let check_idx = check_pos.to_one_dim(cap);

                // DEBUG PRINT
                if (visible)(check_block) {
                    println!(
                        "  > Expanding X to {:?}: HIT (Block: {:?})",
                        check_pos, check_block
                    );
                } else {
                    println!(
                        "  > Expanding X to {:?}: MISS (Block: {:?})",
                        check_pos, check_block
                    );
                }

                if visited[check_idx] || !(visible)(check_block) || check_block != start_block {
                    break;
                }
                size_z += 1;
            }

            // 2. Expand Y
            let mut size_y = 0;
            'y_loop: while pos[1] + size_y < cap[1] {
                for z in 0..size_z {
                    let check_pos = pos + Vector([0, size_y, z]);
                    let check_block = vol.get(check_pos);
                    let check_idx = check_pos.to_one_dim(cap);

                    if visited[check_idx] || !(visible)(check_block) || check_block != start_block {
                        break 'y_loop;
                    }
                }
                size_y += 1;
            }

            // 3. Expand Z
            let mut size_x = 0;
            'z_loop: while pos[2] + size_x < cap[2] {
                for y in 0..size_y {
                    for z in 0..size_z {
                        let check_pos = pos + Vector([size_x, y, z]);
                        let check_block = vol.get(check_pos);
                        let check_idx = check_pos.to_one_dim(cap);

                        if visited[check_idx]
                            || !(visible)(check_block)
                            || check_block != start_block
                        {
                            break 'z_loop;
                        }
                    }
                }
                size_x += 1;
            }

            println!("Found Size: {:?}", [size_x,size_y,size_z]);


            // Mark visited
            for z in 0..size_z {
                for y in 0..size_y {
                    for x in 0..size_x {
                        let p = pos + Vector([x, y, z]);
                        visited[p.to_one_dim(cap)] = true;
                    }
                }
            }

            ret.push(BoundingBox {
                min: pos,
                size: Vector([size_x, size_y, size_z]),
                block: start_block.clone(),
            });
        }
        ret
    }
}
