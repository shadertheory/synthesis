#![feature(trait_alias, thread_local, inherent_associated_types)]

use std::{
    cell::RefCell,
    convert::identity,
    mem::{self, MaybeUninit},
    ops::{BitAnd, BitOr, BitOrAssign, Deref, Index, IndexMut, Shl, Shr},
    slice,
};

#[thread_local]
pub static INSTANCE: RefCell<Option<Platform>> = RefCell::new(None);

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
    type Volume: Volume;
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
pub extern "C" fn engine_start() {
    INSTANCE.replace(Some(unsafe { Platform::init() }));
}

#[unsafe(no_mangle)]
pub extern "C" fn engine_draw() {
    INSTANCE.try_borrow().unwrap().as_ref().unwrap().draw();
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Block(usize);

pub mod darwin {
    use std::rc::Rc;
    use std::time::Duration;
    use std::{cell::RefCell, ptr::NonNull};
    use std::{mem, thread};

    use objc2::{rc::*, runtime::*, *};
    use objc2_core_graphics::*;
    use objc2_foundation::*;
    use objc2_metal::*;
    use objc2_metal_kit::*;
    use objc2_quartz_core::*;
    use objc2_ui_kit::*;

    pub trait PlatformDrawable = CAMetalDrawable;

    // iOS implementation using UIView + CAMetalLayer
    #[cfg(target_os = "ios")]
    pub struct Surface {
        view_controller: Retained<UIViewController>,
        screen: Retained<UIScreen>,
        drawable: Option<Retained<ProtocolObject<dyn PlatformDrawable>>>,
        metal_view: Retained<UIView>,
        metal_layer: Retained<CAMetalLayer>,
        metal_device: Retained<ProtocolObject<dyn MTLDevice>>,
        bounds: NSRect,
    }

    #[cfg(target_os = "ios")]
    impl Surface {
        pub unsafe fn new() -> Surface {
            let view_controller: Retained<UIViewController> =
                msg_send_id![objc2::class!(UIViewController), new];

            // Get main screen bounds
            let screen: Retained<UIScreen> = msg_send_id![objc2::class!(UIScreen), mainScreen];
            let bounds: NSRect = msg_send![&*screen, bounds];

            // Create Metal device
            let metal_device = MTLCreateSystemDefaultDevice().expect("Failed to get metal device.");

            // Create UIView
            let metal_view: Retained<UIView> = msg_send_id![
                msg_send_id![UIView::class(), alloc],
                initWithFrame: bounds
            ];

            // Create CAMetalLayer
            let metal_layer: Retained<CAMetalLayer> = msg_send_id![CAMetalLayer::class(), new];

            // Configure the layer
            let _: () = msg_send![&*metal_layer, setDevice: &*metal_device];
            let _: () =
                msg_send![&*metal_layer, setPixelFormat: objc2_metal::MTLPixelFormat::BGRA8Unorm];
            let _: () = msg_send![&*metal_layer, setFramebufferOnly: false];

            // Set the layer's frame to match the view bounds
            let _: () = msg_send![&*metal_layer, setFrame: bounds];

            // Get the view's layer and add metal layer as sublayer
            let view_layer: Retained<CALayer> = msg_send_id![&*metal_view, layer];
            let _: () = msg_send![&*view_layer, addSublayer: &*metal_layer];

            // Set view on view controller
            let _: () = msg_send![&*view_controller, setView: &*metal_view];

            let mut surface = Self {
                drawable: None,
                view_controller,
                screen,
                metal_view,
                metal_layer,
                metal_device,
                bounds,
            };

            // Set initial scale
            surface.set_scale();

            surface
        }
        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.metal_device.clone()
        }
        pub unsafe fn set_scale(&mut self) {
            let native_scale: f64 = msg_send![&*self.screen, nativeScale];
            let _: () = msg_send![&*self.metal_view, setContentScaleFactor: native_scale];
            let _: () = msg_send![&*self.metal_layer, setContentsScale: native_scale];

            // Update drawable size
            let drawable_width = self.bounds.size.width * native_scale;
            let drawable_height = self.bounds.size.height * native_scale;

            let drawable_size = Size {
                width: drawable_width,
                height: drawable_height,
            };
            let _: () = msg_send![&*self.metal_layer, setDrawableSize: drawable_size];
        }
        pub unsafe fn next(&mut self) -> (Retained<ProtocolObject<dyn MTLTexture>>, Size) {
            let drawable: Option<Retained<ProtocolObject<dyn PlatformDrawable>>> =
                msg_send![&self.metal_layer, nextDrawable];
            let Some(texture) = drawable.as_ref().map(|x| x.texture()) else {
                panic!("could not get next texture to present.");
            };
            println!("{:?}", texture);
            self.drawable = drawable;
            let size = self.metal_layer.drawableSize();
            (texture, Size::new(size.width, size.height))
        }

        pub unsafe fn draw(
            &mut self,
            metal: &mut Metal,
            commands: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        ) {
            let Some(drawable) = self.drawable.take() else {
                panic!("No drawable available to present");
            };
            let _: () = msg_send![&metal.command_queue, waitForDrawable:&*drawable];

            metal
                .command_queue
                .commit_count(NonNull::from_ref(&NonNull::from_ref(&*commands).cast()), 1);

            let _: () = msg_send![&metal.command_queue, signalDrawable:&*drawable];

            let _: () = msg_send![&*drawable, present];
        }
    }

    #[cfg(target_os = "macos")]
    pub struct Surface {
        view_controller: Retained<UIViewController>,
        screen: Retained<UIScreen>,
        metal_view: Retained<MTKView>,
        metal_device: Retained<ProtocolObject<dyn MTLDevice>>,
        bounds: Size,
    }
    #[cfg(target_os = "macos")]
    impl Surface {
        pub unsafe fn new() -> Surface {
            let view_controller: Retained<UIViewController> =
                msg_send_id![objc2::class!(UIViewController), new];

            // Get main screen bounds
            let screen: Retained<UIScreen> = msg_send_id![objc2::class!(UIScreen), mainScreen];
            let bounds: Size = msg_send![&*screen, bounds];

            // Create view
            let metal_device: Option<Retained<ProtocolObject<dyn MTLDevice>>> =
                msg_send_id![objc2::class!(MTLDevice), defaultDevice];
            let metal_device = metal_device.expect("No Metal device available");

            let metal_view: Retained<MTKView> = msg_send_id![
                msg_send_id![MTKView::class(), alloc],
                initWithFrame: bounds,
                device: &*metal_device
            ];

            // Configure MTKView
            // Enable vsync (default is YES but being explicit)
            let _: () = msg_send![&*metal_view, setEnableSetNeedsDisplay: false];
            let _: () = msg_send![&*metal_view, setPaused: false];

            // Set view on view controller
            let _: () = msg_send![&*view_controller, setView: &*metal_view];
            Self {
                view_controller,
                screen,
                metal_view,
                metal_device,
                bounds,
            }
        }

        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.metal_device.clone()
        }

        unsafe fn set_scale(&self) {
            let native_scale: f64 = msg_send![&*self.screen, nativeScale];
            let _: () = msg_send![&*self.metal_view, setContentScaleFactor: native_scale];
        }
    }
    pub type Size = NSSize;

    pub struct Window {
        surface: Surface,
    }
    impl Window {
        pub unsafe fn new() -> Self {
            let mut surface = Surface::new();
            surface.set_scale();
            Self { surface }
        }
        pub fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.surface.device()
        }
        pub fn next(&mut self) -> (MTLResourceID, Size) {
            let (texture, size) = unsafe { self.surface.next() };
            (texture.gpuResourceID(), size)
        }
        pub fn draw(
            &mut self,
            metal: &mut Metal,
            commands: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        ) {
            unsafe { self.surface.draw(metal, commands) }
        }
    }

    pub struct Metal {
        frame: usize,
        next_texture: Box<dyn Fn() -> (MTLResourceID, Size)>,
        next_present:
            Option<Box<dyn Fn(&mut Metal, Retained<ProtocolObject<dyn MTL4CommandBuffer>>)>>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
        command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
        frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        ui_raytrace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    }

    impl Metal {
        unsafe fn init(
            device: Retained<ProtocolObject<dyn MTLDevice>>,
            next_texture: Box<dyn Fn() -> (MTLResourceID, Size)>,
            next_present: Box<dyn Fn(&mut Metal, Retained<ProtocolObject<dyn MTL4CommandBuffer>>)>,
        ) -> Self {
            let next_present = Some(next_present);

            if !device.supportsFamily(MTLGPUFamily::Metal4) {
                panic!("Metal 4 not supported.");
            }

            let command_queue = msg_send![&device, newMTL4CommandQueue];
            let command_allocator = device
                .newCommandAllocator()
                .expect("failed to create command allocator");

            let source = include_str!("ui3d.msl");

            let source = NSString::from_str(source);

            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .expect("Failed to compile shader");

            let func_name = NSString::from_str("reference_grid");
            let function = library
                .newFunctionWithName(&func_name)
                .expect("Function not found");

            let ui_raytrace = device
                .newComputePipelineStateWithFunction_error(&function)
                .expect("Failed to create pipeline");

            let frame_event = device
                .newSharedEvent()
                .expect("Could not create shared event");
            let _: () = msg_send![&frame_event, setSignaledValue: 0u64];

            let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
            argument_table_descriptor.setMaxTextureBindCount(1);

            let argument_table = device
                .newArgumentTableWithDescriptor_error(&argument_table_descriptor)
                .expect("failed to make argument table");

            Self {
                frame: 0,
                next_texture,
                next_present,
                device,
                command_queue,
                command_allocator,
                library,
                frame_event,
                argument_table,
                ui_raytrace,
            }
        }

        unsafe fn render(&mut self) {
            if self.frame > 0 {
                let _: bool = msg_send![&self.frame_event, waitUntilSignaledValue: self.frame - 1, timeoutMS: 0u64];
            }

            self.frame += 1;

            let command_buffer = self
                .device
                .newCommandBuffer()
                .expect("failed to create command buffer");
            command_buffer.beginCommandBufferWithAllocator(&self.command_allocator);
            let (id, size) = (self.next_texture)();

            let encoder = command_buffer
                .computeCommandEncoder()
                .expect("failed to make compute encoder");

            encoder.setComputePipelineState(&self.ui_raytrace);
            self.argument_table.setTexture_atIndex(id, 0);
            encoder.setArgumentTable(Some(&self.argument_table));
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: size.width as usize / 16 + 1,
                    height: size.height as usize / 16 + 1,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
            println!(
                "{:?} {:?}",
                MTLSize {
                    width: size.width as usize / 16 + 1,
                    height: size.height as usize / 16 + 1,
                    depth: 1
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1
                }
            );

            encoder.endEncoding();
            command_buffer.endCommandBuffer();

            let mut present = self.next_present.take();
            (present.as_mut().unwrap())(self, command_buffer);
            self.next_present = present;
            println!("presenting!");
            thread::sleep(Duration::from_secs_f32(1.0 / 120.0));
        }
    }

    pub struct Darwin {
        metal: RefCell<Metal>,
    }

    impl Darwin {
        pub unsafe fn init() -> Self {
            let window = Window::new();
            let window = Rc::new(RefCell::new(window));
            let draw = window.clone();
            let present = window.clone();
            let metal = Metal::init(
                window.borrow_mut().device(),
                Box::new(move || draw.borrow_mut().next()),
                Box::new(move |metal, commands| present.borrow_mut().draw(metal, commands)),
            );
            let metal = RefCell::new(metal);
            Self { metal }
        }

        pub fn draw(&self) {
            unsafe { self.metal.borrow_mut().render() }
        }
    }

    impl super::World for Darwin {
        type Object = Structure;
    }
    impl super::Graphics for Darwin {
        type Renderer = Renderer;

        fn renderer_of(obj: <Self as super::World>::Object) -> <Self as super::Graphics>::Renderer {
            todo!()
        }
    }
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    pub struct Structure(usize);

    pub struct Renderer();

    impl super::Renderer for Renderer {
        fn hide() {
            todo!()
        }

        fn show() {
            todo!()
        }
    }

    pub struct Context {}
}

pub trait Volume {
    fn get(&self, position: Vector<3, usize>) -> Block;
    fn set(&self, position: Vector<3, usize>, block: Block);
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

pub trait Tensor {
    type Repr;
}

#[derive(Clone, Copy)]
pub struct Vector<const DIM: usize, T>([T; DIM]);

impl<const DIM: usize, T> Tensor for Vector<DIM, T> {
    type Repr = T;
}

impl<const DIM: usize, T> Index<usize> for Vector<DIM, T> {
    type Output = T;
    fn index(&self, index: usize) -> &<Self as Index<usize>>::Output {
        &self.0[index]
    }
}

impl<const DIM: usize, T> IndexMut<usize> for Vector<DIM, T> {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        &mut self.0[index]
    }
}

pub trait Zero {
    fn zero() -> Self
    where
        Self: Sized;
}
pub trait One {
    fn one() -> Self
    where
        Self: Sized;
}

macro_rules! primitive {
    ($type:tt) => {
        impl Zero for $type {
            fn zero() -> Self
            where
                Self: Sized,
            {
                0
            }
        }

        impl One for $type {
            fn one() -> Self
            where
                Self: Sized,
            {
                1
            }
        }
    };
}
primitive!(usize);
primitive!(u32);

pub trait Num = Copy + Zero;

impl<const DIM: usize, T: Num> Default for Vector<DIM, T> {
    fn default() -> Self {
        let mut ret = unsafe { mem::zeroed::<[T; DIM]>() };
        for i in 0..DIM {
            ret[i] = T::zero();
        }
        Self(ret)
    }
}

macro_rules! vector_op {
    ($trait:tt, $func:tt, $op:tt) => {
        use std::ops::$trait;
        impl<const DIM: usize, T: $trait + Num> $trait<Self> for Vector<DIM, T> where <T as $trait>::Output: Num {
            type Output = Vector<DIM, <T as $trait>::Output>;
            fn $func(self, rhs: Self) -> <Self as $trait<Self>>::Output {
                let mut ret = <Self as $trait<Self>>::Output::default();
                for i in 0..DIM {
                    ret[i] = self[i] $op rhs[i];
                }
                ret
            }
        }
        impl<const DIM: usize, T: $trait + Num> $trait<T> for Vector<DIM, T> where <T as $trait>::Output: Num {
            type Output = Vector<DIM, <T as $trait>::Output>;
            fn $func(self, rhs: T) -> <Self as $trait<T>>::Output {
                let mut ret = <Self as $trait<T>>::Output::default();
                for i in 0..DIM {
                    ret[i] = self[i] $op rhs;
                }
                ret
            }
        }
    };
}

vector_op!(Add, add, +);
vector_op!(Sub, sub, -);
vector_op!(Mul, mul, *);
vector_op!(Div, div, /);
vector_op!(Rem, rem, %);

pub trait IndexExt: Tensor {
    fn to_one_dim(self, size: usize) -> <Self as Tensor>::Repr;
    fn from_one_dim(idx: <Self as Tensor>::Repr, size: usize) -> Self;
}

impl<const DIM: usize, T> IndexExt for Vector<DIM, T> {
    fn to_one_dim(self, size: usize) -> <Self as Tensor>::Repr {
        todo!()
    }

    fn from_one_dim(idx: <Self as Tensor>::Repr, size: usize) -> Self {
        todo!()
    }
}

pub const SIZE: usize = 8;
pub const NOMINAL: usize = SIZE * SIZE * SIZE;

pub struct Region {
    data: RefCell<[Box<Chunk>; NOMINAL]>,
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

pub struct Palette<Repr: PaletteRepr = u32> {
    palette: Array<Block>,
    len: usize,
    compressed: Vec<Repr>,
}

impl<Repr: PaletteRepr> Palette<Repr> {
    const BYTES: usize = mem::size_of::<Repr>();
    const BITS: usize = 8 * Self::BYTES;
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

        let mut repr = Repr::zero();

        let one = Repr::one();
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
    fn compress(blocks: &[Block]) -> Self {
        let mut palette = blocks.to_vec();
        palette.dedup_by_key(|x| x.0);
        let palette = Array::<Block>::from(palette);

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

    fn decompress(&self) -> Vec<Block> {
        let Self {
            palette,
            len,
            compressed,
        } = self;
        let mut ret = vec![];
        let size = palette.len();
        for place in 0..*len {
            let index = Self::read(&compressed, place, size);
            let block = palette[index.try_into().ok().unwrap()];
            ret.push(block);
        }
        ret
    }

    fn get(&self, index: usize) -> Block {
        let palette_index = Self::read(&*self.compressed, index, self.palette.len());
        self.palette[palette_index.try_into().ok().unwrap()]
    }

    fn set(&mut self, index: usize, block: Block) {
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

    fn contains(&self, block: Block) -> bool {
        self.palettize(block).is_some()
    }

    fn palettize(&self, block: Block) -> Option<&Block> {
        if let Some(idx) = self.palette.binary_search(&block).ok() {
            Some(&self.palette[idx])
        } else {
            None
        }
    }
}

pub struct Chunk {
    palette: Palette,
}

impl Index<usize> for Chunk {
    type Output = Block;

    fn index(&self, index: usize) -> &<Self as Index<usize>>::Output {
        self.get(index)
    }
}

impl Chunk {
    fn set(&mut self, index: usize, block: Block) {
        if self.palette.contains(block) {
            let mut blocks = self.palette.decompress();
            blocks[index] = block;
            self.palette = Palette::compress(&*blocks);
        } else {
            self.palette.set(index, block);
        }
    }

    fn get(&self, index: usize) -> &Block {
        self.palette.palettize(self.palette.get(index)).unwrap()
    }
}

impl Volume for Region {
    fn get(&self, position: Vector<3, usize>) -> Block {
        let local = (position % SIZE).to_one_dim(SIZE);
        let global = (position / SIZE).to_one_dim(SIZE);
        self.data.borrow()[global][local]
    }

    fn set(&self, position: Vector<3, usize>, block: Block) {
        let local = (position % SIZE).to_one_dim(SIZE);
        let global = (position / SIZE).to_one_dim(SIZE);
        self.data.borrow_mut()[global].set(local, block);
    }

    fn apply(&self, brush: Brush) {}
}
