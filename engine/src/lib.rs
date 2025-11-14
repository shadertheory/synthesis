#![feature(
    trait_alias,
    thread_local,
    inherent_associated_types,
    generic_const_exprs,
    associated_type_defaults,
    ptr_metadata,
    box_as_ptr
)]

mod darwin;
mod math;
mod world;

pub use math::*;

use std::{
    cell::{RefCell, UnsafeCell},
    collections::BTreeMap,
    convert::identity,
    fmt,
    iter::{self, Sum},
    mem::{self, MaybeUninit},
    ops::{BitAnd, BitOr, BitOrAssign, Deref, Index, IndexMut, Neg, Shl, Shr, *},
    slice,
};

#[thread_local]
pub static mut INSTANCE: Option<Platform> = None;

type Inline<T, const CAP: usize> = [MaybeUninit<T>; CAP];
pub enum Array<T, const CAP: usize = 16> {
    Inline(usize, Inline<T, CAP>),
    Overflow(Vec<T>),
}
impl<T, const CAP: usize> Array<T, CAP> {}

impl<T, const CAP: usize> Deref for Array<T, CAP> {
    type Target = [T];
    fn deref(&self) -> &<Self as Deref>::Target {
        use Array::*;
        match self {
            Inline(_, arr) => unsafe { arr.as_ptr().cast::<[T; CAP]>().as_ref().unwrap() },
            Overflow(vec) => &*vec,
        }
    }
}

impl<T, const CAP: usize> From<Vec<T>> for Array<T, CAP> {
    fn from(value: Vec<T>) -> Self {
        if value.len() < CAP {
            let mut ret = unsafe { mem::zeroed::<Inline<T, CAP>>() };
            let len = value.len();
            for (i, val) in value.into_iter().enumerate() {
                ret[i].write(val);
            }
            return Self::Inline(len, ret);
        };
        Self::Overflow(value)
    }
}

type Platform = darwin::Darwin;

pub trait Engine = World + Voxel + Graphics + Physics;

pub trait World {
    type Object: Ord;
}

pub trait Voxel: World {
    type Volume: Volume<3, Region>;
    fn volume_of(obj: <Self as World>::Object) -> Self::Volume;
}

pub trait Graphics: World {
    type Renderer: Renderer;
    fn renderer_of(obj: <Self as World>::Object) -> Self::Renderer;
}

pub trait Physics: World {
    type Rigidbody: Rigidbody;
    fn rigidbody_of(obj: <Self as World>::Object) -> Self::Rigidbody;
}

#[unsafe(no_mangle)]
pub extern "C" fn engine_start(data: *const ()) {
    unsafe {
        INSTANCE = Some(Platform::init(data));
    };
}

#[unsafe(no_mangle)]
pub extern "C" fn engine_draw() {
    unsafe {
        #[allow(static_mut_refs)]
        INSTANCE.as_mut().unwrap().draw();
    }
}

pub trait Field {}

pub enum BrushOperation {
    Paint,
}

pub struct Brush {
    field: Box<dyn Field>,
    op: BrushOperation,
    block: Block,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CameraData {
    projection: Matrix<4, 4, f32>,
    projection_inverse: Matrix<4, 4, f32>,
    view: Matrix<4, 4, f32>,
    transform: Matrix<4, 4, f32>,
    resolution: Vector<4, f32>,
}

impl CameraData {
    fn unproject(
        &self,
        points: Vector<2, f32>,
        size: Vector<2, f32>,
        scale: f32,
    ) -> Raycast<3, f32> {
        // Convert to NDC space
        let ndc = Vector([
            points[0] / size[0] * 2.0 - 1.0,
            points[1] / size[1] * 2.0 - 1.0,
            1.0,
            1.0,
        ]);

        // Unproject to view space
        let view_pos = self.projection_inverse * ndc;

        // Perspective divide to get view direction
        let view_dir = Vector([
            view_pos[0] / view_pos[3] * self.projection_inverse[(1, 1)],
            view_pos[1] / view_pos[3] * self.projection_inverse[(0, 0)],
            -view_pos[2] / view_pos[3],
            0.0, // Direction vector (w=0)
        ]);

        // Transform to world space (direction remains unnormalized initially)
        let world_dir = self.transform * view_dir;
        let world_dir = Vector([world_dir[0], world_dir[1], world_dir[2]]);

        // Extract camera position from transform matrix
        let camera_pos = Vector([
            self.transform[(0, 3)],
            self.transform[(1, 3)],
            self.transform[(2, 3)],
        ]);

        dbg!(ndc, view_dir, world_dir, camera_pos);

        Raycast {
            origin: camera_pos,
            direction: world_dir.normalize(),
        }
    }
}

pub enum Input {
    Pan(Vector<2, f32>, Vector<2, f32>),
    Zoom(f32),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Block(usize);

pub static mut STRUCTURE: Option<BTreeMap<Block, Palette<Cell, u32>>> = None;

fn init_structure() {
    let mut structure = BTreeMap::<Block, Palette<Cell, u32>>::new();
    structure.insert(Block(0), Palette::new(Cell::Vacuum, 512));
    structure.insert(
        Block(1),
        Palette::compress(
            (0..)
                .map(|i| match i % 2 {
                    0 => Cell::Vacuum,
                    1 => Cell::Stone,
                    _ => unreachable!(),
                })
                .take(512)
                .collect::<Vec<Cell>>()
                .as_slice(),
        ),
    );
    unsafe {
        STRUCTURE = Some(structure);
    }
}

pub trait Volume<const DIM: usize, Subtype> {
    fn get(&self, position: Vector<DIM, usize>) -> &Subtype;
    fn set(&self, position: Vector<DIM, usize>, block: Subtype);
    fn apply(&self, brush: Brush);
}

pub trait Renderer {
    fn hide();
    fn show();
}

pub trait Rigidbody {
    fn force();
    fn torque();
}

pub const SIZE: usize = 8;
pub const NOMINAL: usize = SIZE * SIZE;

pub struct Region {
    data: UnsafeCell<[Box<Chunk>; NOMINAL]>,
}

pub trait PaletteRepr = BitOr
    + BitOrAssign
    + BitAnd
    + Shl<usize>
    + Shr<usize>
    + Sub
    + Copy
    + Zero
    + One
    + TryFrom<usize>
    + TryInto<usize>
where
    <Self as Shl<usize>>::Output: Into<Self>,
    <Self as Shr<usize>>::Output: Into<Self>,
    <Self as Sub>::Output: Into<Self>,
    <Self as BitOr>::Output: Into<Self>,
    <Self as BitAnd>::Output: Into<Self>;

pub struct Palette<Compress, Repr: PaletteRepr = u32> {
    palette: Array<Compress>,
    len: usize,
    compressed: Vec<Repr>,
}

impl<Compress> Index<usize> for Palette<Compress>
where
    Compress: Ord + Clone,
{
    type Output = Compress;
    fn index(&self, index: usize) -> &<Self as Index<usize>>::Output {
        self.palettize(self.get(index)).unwrap()
    }
}

impl<Compress, Repr: PaletteRepr> Palette<Compress, Repr> {
    const BYTES: usize = mem::size_of::<Repr>();
    const BITS: usize = 8 * Self::BYTES;

    pub fn new(repr: Compress, count: usize) -> Self
    where
        Compress: Clone + Ord,
    {
        Self::compress(iter::repeat_n(repr, count).collect::<Vec<_>>().as_slice())
    }
    fn calculate_bits(palette_len: usize) -> usize {
        (palette_len as f32).log2().ceil() as usize
    }
    fn write(
        buf: &mut [MaybeUninit<Repr>],
        index_placement: usize,
        palette_val: Repr,
        size_palette: usize,
    ) {
        let bits = Self::calculate_bits(size_palette);
        let index_repr = index_placement / Self::BITS;
        let index_chunk = index_placement % Self::BITS;
        let overflow = (index_chunk + 1) * bits >= Self::BITS;

        let current = unsafe { buf[index_repr].assume_init() };
        let shift = (bits * index_chunk);
        let new = (palette_val << shift).into();
        buf[index_repr].write((current | new).into());

        if overflow {
            let current = unsafe { buf[index_repr + 1].assume_init() };
            let shift_overflow = bits - ((bits + 1) * index_chunk - Self::BITS);
            let new = (palette_val >> shift_overflow).into();
            buf[index_repr + 1].write((current | new).into());
        }
    }
    fn read(buf: &[Repr], index_placement: usize, size_palette: usize) -> Repr {
        let bits = Self::calculate_bits(size_palette);
        let index_repr = index_placement / Self::BITS;
        let index_chunk = index_placement % Self::BITS;
        let overflow = (index_chunk + 1) * bits >= Self::BITS;

        let mut repr = Repr::ZERO;

        let one = Repr::ONE;
        let shift = bits * index_chunk;
        let shift_underflow = ((bits + 1) * index_chunk - Self::BITS);
        let shift_overflow = bits - shift_underflow;
        let mask = ((one << bits.try_into().unwrap()).into() - one).into();
        let val = (buf[index_repr] >> shift).into();
        let mask_shift = (mask >> shift_overflow).into();
        repr |= (val & mask_shift).into();

        if overflow {
            let val = buf[index_repr];
            let mask_shift = (mask >> shift_underflow).into();
            let val_masked = (val & mask_shift).into();
            repr |= (val_masked << shift_underflow).into();
        }

        repr
    }
    fn compress(blocks: &[Compress]) -> Self
    where
        Compress: Clone + PartialEq + Ord,
    {
        let mut palette = blocks.to_vec();
        palette.dedup();
        let palette = Array::<Compress>::from(palette);

        let len = blocks.len();
        let size = palette.len();
        let bits = Self::calculate_bits(size);

        let mut compressed: Vec<MaybeUninit<Repr>> = vec![];
        unsafe {
            compressed.set_len(len * bits);
        }

        for place in 0..len {
            let val = palette
                .binary_search(&blocks[place])
                .unwrap()
                .try_into()
                .ok()
                .unwrap();

            Self::write(&mut compressed, place, val, size);
        }
        let compressed = unsafe { mem::transmute(compressed) };

        Self {
            palette,
            len,
            compressed,
        }
    }

    fn decompress(&self) -> Vec<Compress>
    where
        Compress: Clone,
    {
        let Self {
            palette,
            len,
            compressed,
        } = self;
        let mut ret = vec![];
        let size = palette.len();
        for place in 0..*len {
            let index = Self::read(&compressed, place, size);
            let block = palette[index.try_into().ok().unwrap()].clone();
            ret.push(block);
        }
        ret
    }

    fn get(&self, index: usize) -> Compress
    where
        Compress: Clone + Ord,
    {
        let palette_index = Self::read(&*self.compressed, index, self.palette.len());
        self.palette[palette_index.try_into().ok().unwrap()].clone()
    }

    fn set(&mut self, index: usize, block: Compress)
    where
        Compress: Ord,
    {
        let val = self
            .palette
            .binary_search(&block)
            .unwrap()
            .try_into()
            .ok()
            .unwrap();

        let compressed = unsafe {
            slice::from_raw_parts_mut(
                self.compressed.as_mut_ptr().cast::<MaybeUninit<Repr>>(),
                self.compressed.len(),
            )
        };

        Self::write(compressed, index, val, self.palette.len());
    }

    fn contains(&self, block: Compress) -> bool
    where
        Compress: Ord,
    {
        self.palettize(block).is_some()
    }

    fn palettize(&self, block: Compress) -> Option<&Compress>
    where
        Compress: Ord,
    {
        if let Some(idx) = self.palette.binary_search(&block).ok() {
            Some(&self.palette[idx])
        } else {
            None
        }
    }
}

pub struct Chunk {
    palette: UnsafeCell<Palette<Block>>,
}

impl Index<usize> for Chunk {
    type Output = Block;

    fn index(&self, index: usize) -> &<Self as Index<usize>>::Output {
        self.get(Vector::from_one_dim(index, 8))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Cell {
    Vacuum,
    Grass,
    Stone,
    Snow,
    Water,
    Dirt,
    Leaf,
    Bark,
    Wood { dark: bool },
}

impl Volume<3, Cell> for Block {
    fn get(&self, position: Vector<3, usize>) -> &Cell {
        #[allow(static_mut_refs)]
        let structure = unsafe { crate::STRUCTURE.as_mut().unwrap() };
        &structure[self].palette[position.to_one_dim(8)]
    }

    fn set(&self, position: Vector<3, usize>, cell: Cell) {
        unreachable!();
    }

    fn apply(&self, brush: Brush) {
        todo!()
    }
}

impl Volume<3, Block> for Chunk {
    fn get(&self, index: Vector<3, usize>) -> &Block {
        let palette = unsafe { self.palette.get().as_mut().unwrap() };
        &palette[index.to_one_dim(8)]
    }

    fn set(&self, index: Vector<3, usize>, block: Block) {
        let index = index.to_one_dim(8);
        let palette = unsafe { self.palette.get().as_mut().unwrap() };
        if palette.contains(block) {
            let mut blocks = palette.decompress();
            blocks[index] = block;
            *palette = Palette::compress(&*blocks);
        } else {
            palette.set(index, block);
        }
    }

    fn apply(&self, brush: Brush) {
        todo!()
    }
}

impl<'a> Volume<2, Chunk> for Region {
    fn get(&self, position: Vector<2, usize>) -> &Chunk {
        let global = (position / SIZE).to_one_dim(SIZE);
        unsafe { &*self.data.get().as_ref().unwrap()[global] }
    }

    fn set(&self, position: Vector<2, usize>, chunk: Chunk) {
        let global = (position / SIZE).to_one_dim(SIZE);
        unsafe {
            *self.data.get().as_mut().unwrap()[global] = chunk;
        }
    }

    fn apply(&self, brush: Brush) {}
}
