use core::array;
use core::slice;
use std::f32::EPSILON;
use std::f32::consts::PI;
use std::random;
use std::random::DefaultRandomSource;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use std::{cell::RefCell, ptr::NonNull};
use std::{mem, ptr, thread};

use noise::NoiseFn;
use objc2::{rc::*, runtime::*, *};
use objc2_core_graphics::*;
use objc2_foundation::*;
use objc2_game_controller::GCController;
use objc2_metal::*;
use objc2_metal_kit::*;
use objc2_quartz_core::*;
use objc2_ui_kit::*;
use rand::Rng;

use crate::acoustics::*;
use crate::darwin::rtx::{BoundingBox, GeometryDescriptor, InstanceDescriptor};
use math::*;

pub trait PlatformDrawable = CAMetalDrawable;

// iOS implementation using UIView + CAMetalLayer
#[cfg(target_os = "ios")]
pub struct Surface {
    view_controller: &'static UIViewController,
    screen: Retained<UIScreen>,
    drawable: Option<Retained<ProtocolObject<dyn PlatformDrawable>>>,
    metal_view: Retained<UIView>,
    metal_layer: Retained<CAMetalLayer>,
    metal_device: DeviceProtocol,
    native_scale: f32,
    bounds: NSRect,
}

#[cfg(target_os = "ios")]
impl Surface {
    /// Get UIScreen from UIWindowScene
    pub fn main_screen(app: &UIApplication) -> Retained<UIScreen> {
        app.windows().firstObject().unwrap().screen()
    }

    pub fn size(&self) -> Vector<2, f32> {
        let bounds: NSRect = unsafe { msg_send![&*self.screen, bounds] };
        Vector([bounds.size.width as f32, bounds.size.height as f32])
    }

    pub unsafe fn new(view_controller: &'static UIViewController) -> Surface {
        let app = UIApplication::sharedApplication(MainThreadMarker::new().unwrap());

        // Get main screen bounds
        let screen: Retained<UIScreen> = Self::main_screen(&app);
        let bounds: NSRect = msg_send![&*screen, bounds];
        println!("Got screen with bounds {:?}", bounds);

        // Create UIView
        let metal_view: Retained<UIView> =
            UIView::init(UIView::alloc(MainThreadMarker::new().unwrap()));
        println!("Got screen with bounds {:?}", bounds);

        // Create CAMetalLayer
        let metal_layer: Retained<CAMetalLayer> = CAMetalLayer::init(CAMetalLayer::alloc());
        let metal_device = metal_layer.device().expect("could not get device");
        println!("Got screen with bounds {:?}", bounds);

        // Configure the layer
        let _: () =
            msg_send![&*metal_layer, setPixelFormat: objc2_metal::MTLPixelFormat::BGRA8Unorm];

        // Set the layer's frame to match the view bounds
        let _: () = msg_send![&*metal_layer, setFrame: bounds];
        println!("Got screen with bounds {:?}", bounds);

        // Get the view's layer and add metal layer as sublayer
        metal_view.layer().addSublayer(&metal_layer);

        // Set view on view controller
        view_controller.setView(Some(&metal_view));
        println!("Got screen with bounds {:?}", bounds);

        let mut surface = Self {
            drawable: None,
            native_scale: 1.0,
            view_controller,
            screen,
            metal_view,
            metal_layer,
            metal_device,
            bounds,
        };
        unsafe {
            surface.set_scale();
        }

        surface
    }
    fn device(&self) -> DeviceProtocol {
        self.metal_device.clone()
    }
    pub unsafe fn add_gesture(&mut self, gesture: &UIGestureRecognizer) {
        self.view_controller
            .view()
            .unwrap()
            .addGestureRecognizer(gesture);
    }
    pub unsafe fn set_scale(&mut self) {
        let native_scale: f64 = msg_send![&*self.screen, nativeScale];
        let _: () = msg_send![&*self.metal_view, setContentScaleFactor: native_scale];
        let _: () = msg_send![&*self.metal_layer, setContentsScale: native_scale];
        self.native_scale = native_scale as f32;

        // Update drawable size
        let drawable_width = self.bounds.size.width * native_scale;
        let drawable_height = self.bounds.size.height * native_scale;

        let drawable_size = Size {
            width: drawable_width,
            height: drawable_height,
        };
        let _: () = msg_send![&*self.metal_layer, setDrawableSize: drawable_size];
    }
    pub unsafe fn next(&mut self) -> (TextureProtocol, Size) {
        let drawable: Retained<ProtocolObject<dyn PlatformDrawable>> =
            self.metal_layer.nextDrawable().unwrap();

        let texture = drawable.texture();

        println!("{:?}", texture);
        self.drawable = Some(drawable);
        let size = self.metal_layer.drawableSize();
        (texture, Size::new(size.width, size.height))
    }

    pub unsafe fn draw(&mut self, metal: &mut Metal) {
        let frame_index = metal.frame % metal.frame_in_flight;

        #[allow(static_mut_refs)]
        let commands = LATEST_CMDS.as_ref().expect("failed to get commands");
        let Some(drawable) = self.drawable.take() else {
            panic!("No drawable available to present");
        };
        let _: () = msg_send![&metal.command_queue, waitForDrawable:&*drawable];

        let cmd_ptr = NonNull::from(&**commands);
        let mut cmd_array = [cmd_ptr];
        let cmd_array_ptr = NonNull::new_unchecked(cmd_array.as_mut_ptr());

        metal.command_queue.commit_count(cmd_array_ptr, 1);
        let _: () = msg_send![&metal.command_queue, signalDrawable:&*drawable];

        drawable.present();

        self.drawable = Some(drawable);
    }
}

#[cfg(target_os = "macos")]
pub struct Surface {
    view_controller: UIViewController,
    screen: Retained<UIScreen>,
    metal_view: Retained<MTKView>,
    metal_device: DeviceProtocol,
    bounds: Size,
}
#[cfg(target_os = "macos")]
impl Surface {
    pub unsafe fn new() -> Surface {
        let view_controller: UIViewController = msg_send_id![objc2::class!(UIViewController), new];

        // Get main screen bounds
        let screen: Retained<UIScreen> = msg_send_id![objc2::class!(UIScreen), mainScreen];
        let bounds: Size = msg_send![&*screen, bounds];

        // Create view
        let metal_device: Option<DeviceProtocol> =
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

    fn device(&self) -> DeviceProtocol {
        self.metal_device.clone()
    }

    unsafe fn set_scale(&self) {
        let native_scale: f64 = msg_send![&*self.screen, nativeScale];
        let _: () = msg_send![&*self.metal_view, setContentScaleFactor: native_scale];
    }
}
pub type Size = NSSize;

pub static mut REBUILT: bool = false;
pub static mut LATEST_CMDS: Option<CommandBufferProtocol> = None;

pub struct Window {
    surface: Surface,
    controller: Option<Retained<GCController>>,
    pan: Retained<UIPanGestureRecognizer>,
    pinch: Retained<UIPinchGestureRecognizer>,
    rotate: Retained<UIRotationGestureRecognizer>,
    inputs: Vec<Input>,
}
impl Window {
    pub fn screen_size(&self) -> Vector<2, f32> {
        self.surface.size()
    }
    pub fn native_scale(&self) -> f32 {
        self.surface.native_scale
    }
    pub unsafe fn new(view_controller: &'static UIViewController) -> Self {
        let mut surface = Surface::new(view_controller);

        //gestures
        let pan = UIPanGestureRecognizer::init(UIPanGestureRecognizer::alloc(
            MainThreadMarker::new().unwrap(),
        ));
        let pinch = UIPinchGestureRecognizer::init(UIPinchGestureRecognizer::alloc(
            MainThreadMarker::new().unwrap(),
        ));
        let rotate = UIRotationGestureRecognizer::init(UIRotationGestureRecognizer::alloc(
            MainThreadMarker::new().unwrap(),
        ));
        unsafe {
            surface.add_gesture(&pan);
            surface.add_gesture(&pinch);
            surface.add_gesture(&rotate);
        }
        let inputs = Vec::with_capacity(1024);

        Self {
            controller: None,
            surface,
            pan,
            pinch,
            rotate,
            inputs,
        }
    }
    pub fn device(&self) -> DeviceProtocol {
        self.surface.device()
    }
    pub fn next(&mut self) -> (TextureProtocol, Size) {
        let (texture, size) = unsafe { self.surface.next() };
        (texture, size)
    }
    unsafe fn update_rotate(&mut self) {
        let state: UIGestureRecognizerState = msg_send![&*self.rotate, state];

        let velocity: f32 = match state {
            UIGestureRecognizerState::Changed | UIGestureRecognizerState::Ended => {
                self.rotate.velocity() as f32
            }
            _ => return,
        };

        self.inputs.push(Input::Rotate(-velocity));
    }
    unsafe fn update_pan(&mut self) {
        let state: UIGestureRecognizerState = msg_send![&*self.pan, state];
        let location: NSPoint =
            msg_send![&*self.pan, locationInView: Some(&*self.surface.metal_view)];

        let velocity: NSPoint = match state {
            UIGestureRecognizerState::Changed => {
                msg_send![
                    &*self.pan,
                    velocityInView: Some(&*self.surface.metal_view)
                ]
            }
            UIGestureRecognizerState::Ended => {
                msg_send![
                    &*self.pan,
                    velocityInView: Some(&*self.surface.metal_view)
                ]
            }
            _ => return,
        };

        self.inputs
            .push(Input::Pan(location.into(), velocity.into()));
    }
    unsafe fn update_pinch(&mut self) {
        let state: UIGestureRecognizerState = msg_send![&*self.pinch, state];
        let velocity: f32 = match state {
            UIGestureRecognizerState::Changed | UIGestureRecognizerState::Ended => {
                self.pinch.velocity() as f32
            }
            _ => return,
        };
        self.inputs.push(Input::Zoom(velocity));
    }
    pub unsafe fn update_gamepad(&mut self) {
        let Some(gamepad) = self.controller.as_mut() else {
            self.controller = GCController::current();
            return;
        };

        let Some(gamepad) = gamepad.extendedGamepad() else {
            return;
        };

        let left_stick = gamepad.leftThumbstick();
        self.inputs.push(Input::Move(Vector([
            -left_stick.xAxis().value(),
            left_stick.yAxis().value(),
        ])));
        let right_stick = gamepad.rightThumbstick();

        let zoom_pot = right_stick.yAxis().value();

        if zoom_pot.abs() >= EPSILON {
            self.inputs.push(Input::Zoom(-zoom_pot));
        }

        let rot_pot = right_stick.xAxis().value();

        if rot_pot.abs() >= EPSILON {
            self.inputs.push(Input::Rotate(rot_pot));
        }
    }
    pub fn poll_input(&mut self) {
        unsafe {
            self.update_gamepad();
            self.update_pan();
            self.update_rotate();
            self.update_pinch();
        }
    }
    pub fn take_input(&mut self) -> Vec<Input> {
        self.poll_input();
        mem::take(&mut self.inputs)
    }
    pub fn draw(&mut self, metal: &mut Metal) {
        unsafe { self.surface.draw(metal) }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CameraData {
    projection: Matrix<4, 4, f32>,
    projection_inverse: Matrix<4, 4, f32>,
    view: Matrix<4, 4, f32>,
    transform: Matrix<4, 4, f32>,
    resolution: Vector<4, f32>,
    info: Vector<4, u32>,
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
            direction: -world_dir.normalize(),
        }
    }
}

pub enum Input {
    Pan(Vector<2, f32>, Vector<2, f32>),
    Move(Vector<2, f32>),
    Rotate(f32),
    Zoom(f32),
}
mod rtx {
    use std::{
        collections::{BTreeSet, HashMap},
        hash::Hash,
        mem,
        ops::Bound,
        ptr::NonNull,
    };

    use crate::darwin::{
        BufferProtocol, CommandBufferProtocol, DeviceProtocol, FunctionProtocol, LibraryProtocol,
        ResidencyProtocol,
    };
    use math::*;

    use objc2::{AnyThread, msg_send, rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::{NSArray, NSString};
    use objc2_metal::*;
    pub type AccelerationStructureProtocol = Retained<ProtocolObject<dyn MTLAccelerationStructure>>;

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Geometry(usize);
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Instance(usize);
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct BoundingBox {
        pub min: [f32; 3],
        pub max: [f32; 3],
    }

    #[derive(Clone)]
    pub struct IntersectionFunction {
        pub name: String,
    }

    #[derive(Clone)]
    pub struct GeometryDescriptor {
        pub bounding_box: BoundingBox,
        pub func: IntersectionFunction,
        pub opaque: bool,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct PerInstanceData {
        pub transform: Matrix<4, 4, f32>,
        pub inverse_transform: Matrix<4, 4, f32>,
        pub min: Vector<4, f32>,
        pub size: Vector<4, f32>,
    }

    #[derive(Clone, Copy)]
    pub struct InstanceDescriptor {
        pub geometry: Geometry,
        pub rotation: Quaternion<f32>,
        pub position: Vector<3, f32>,
        pub scale: Vector<3, f32>,
        pub mask: u32,
        pub user_id: u32,
    }

    pub enum PendingOp {
        Geometry(Geometry, PendingGeometryOp),
        Instance(Instance, PendingInstanceOp),
    }

    pub enum PendingGeometryOp {
        Add(GeometryDescriptor),
        Remove,
    }

    pub enum PendingInstanceOp {
        Add(InstanceDescriptor),
        Update(InstanceDescriptor),
        Remove,
    }
    pub struct GeometryData {
        primitive: Option<AccelerationStructureProtocol>,
        descriptor: GeometryDescriptor,
        bbox_index: usize,
        dirty: bool,
    }
    pub struct InstanceData {
        descriptor: InstanceDescriptor,
        index: usize,
        dirty: bool,
    }
    pub struct AccelerationStructure {
        device: DeviceProtocol,
        residency: ResidencyProtocol,

        geometry: HashMap<Geometry, GeometryData>,
        geometry_intersector: HashMap<Geometry, usize>,
        next_geometry: usize,

        instance: HashMap<Instance, InstanceData>,
        instance_to_offset: HashMap<Instance, usize>,
        next_instance: usize,

        bbox: BufferProtocol,
        bbox_capacity: usize,
        bbox_used: usize,

        instance_indirect: BufferProtocol,
        instance_indirect_capacity: usize,
        instance_indirect_used: usize,

        per_instance_data: BufferProtocol,
        per_instance_data_capacity: usize,
        per_instance_data_used: usize,

        instance_count: BufferProtocol,

        library: LibraryProtocol,
        linked: Option<Retained<MTLLinkedFunctions>>,
        functions: Option<Vec<FunctionProtocol>>,

        scratch: BufferProtocol,
        scratch_size: usize,
        scratch_offset: usize,

        structure: Option<AccelerationStructureProtocol>,

        pending: Vec<PendingOp>,

        rebuild: bool,
        rebuild_callback: Option<Box<dyn AccelerationStructureRebuild>>,
        resize: bool,
    }

    pub trait AccelerationStructureRebuild = FnMut(&AccelerationStructure) + 'static;

    impl AccelerationStructure {
        const MAX_FUNCTIONS: usize = 16;
        pub fn new(
            device: &DeviceProtocol,
            residency: &ResidencyProtocol,
            library: &LibraryProtocol,
        ) -> Self {
            Self::with_capacity(device, residency, library, 256, 4096, 16 * 1024 * 1024)
        }
        pub fn per_instance(&self) -> MTLGPUAddress {
            self.per_instance_data.gpuAddress()
        }
        pub fn with_capacity(
            device: &DeviceProtocol,
            residency: &ResidencyProtocol,
            library: &LibraryProtocol,
            geometry_capacity: usize,
            instance_capacity: usize,
            scratch_size: usize,
        ) -> Self {
            let (device, residency, library) = (device.clone(), residency.clone(), library.clone());
            let bbox = device
                .newBufferWithLength_options(
                    instance_capacity * mem::size_of::<BoundingBox>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("failed to make bbox buffer");
            let _: () = unsafe { msg_send![&*residency, addAllocation: &*bbox] };

            let instance_indirect = device
                .newBufferWithLength_options(
                    instance_capacity
                        * mem::size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("failed to make instance indirect buffer");
            let _: () = unsafe { msg_send![&*residency, addAllocation: &*instance_indirect] };

            let per_instance_data = device
                .newBufferWithLength_options(
                    instance_capacity * mem::size_of::<PerInstanceData>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("failed to make per_instance_data buffer");
            let _: () = unsafe { msg_send![&*residency, addAllocation: &*per_instance_data] };

            let instance_count = device
                .newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared)
                .expect("failed to make instance count buffer");
            let _: () = unsafe { msg_send![&*residency, addAllocation: &*instance_count] };

            let scratch = device
                .newBufferWithLength_options(scratch_size, MTLResourceOptions::StorageModePrivate)
                .expect("failed to make scratch buffer");
            let _: () = unsafe { msg_send![&*residency, addAllocation: &*scratch] };

            residency.commit();
            residency.requestResidency();

            Self {
                device,
                residency,
                geometry: Default::default(),
                next_geometry: 0,
                instance: Default::default(),
                instance_to_offset: Default::default(),
                next_instance: 0,
                bbox,
                bbox_capacity: geometry_capacity,
                bbox_used: 0,
                instance_indirect,
                instance_indirect_capacity: instance_capacity,
                instance_indirect_used: 0,
                per_instance_data,
                per_instance_data_capacity: instance_capacity,
                per_instance_data_used: 0,
                instance_count,
                scratch,
                scratch_offset: 0,
                scratch_size,
                library,
                linked: None,
                structure: None,
                pending: vec![],
                functions: None,
                rebuild: false,
                rebuild_callback: None,
                geometry_intersector: Default::default(),
                resize: false,
            }
        }
        pub fn set_rebuild_callback(&mut self, cb: impl AccelerationStructureRebuild) {
            self.rebuild_callback = Some(Box::new(cb));
        }
        pub fn add_geometry(&mut self, descriptor: GeometryDescriptor) -> Geometry {
            let id = Geometry(self.next_geometry);
            self.next_geometry += 1;
            self.pending
                .push(PendingOp::Geometry(id, PendingGeometryOp::Add(descriptor)));
            self.rebuild = true;
            id
        }
        pub fn remove_geometry(&mut self, id: Geometry) {
            self.pending
                .push(PendingOp::Geometry(id, PendingGeometryOp::Remove));
            self.rebuild = true;
        }
        pub fn add_instance(&mut self, descriptor: InstanceDescriptor) -> Instance {
            let id = Instance(self.next_instance);
            self.next_instance += 1;
            self.pending
                .push(PendingOp::Instance(id, PendingInstanceOp::Add(descriptor)));
            self.rebuild = true;
            id
        }
        pub fn remove_instance(&mut self, id: Instance) {
            self.pending
                .push(PendingOp::Instance(id, PendingInstanceOp::Remove));
            self.rebuild = true;
        }
        pub fn update_instance(&mut self, id: Instance, descriptor: InstanceDescriptor) {
            self.pending.push(PendingOp::Instance(
                id,
                PendingInstanceOp::Update(descriptor),
            ));
            self.rebuild = true;
        }
        pub unsafe fn update(&mut self, cmd: &CommandBufferProtocol) {
            let ops = mem::take(&mut self.pending);
            if ops.len() > 0 {
                self.rebuild = true;
            }
            for op in ops {
                match op {
                    PendingOp::Geometry(id, PendingGeometryOp::Add(desc)) => {
                        self.add_geometry_commit(cmd, id, desc)
                    }
                    PendingOp::Geometry(id, PendingGeometryOp::Remove) => {
                        self.remove_geometry_commit(cmd, id)
                    }
                    PendingOp::Instance(id, PendingInstanceOp::Add(desc)) => {
                        self.add_instance_commit(cmd, id, desc)
                    }
                    PendingOp::Instance(id, PendingInstanceOp::Update(desc)) => {
                        self.update_instance_commit(cmd, id, desc)
                    }
                    PendingOp::Instance(id, PendingInstanceOp::Remove) => {
                        self.remove_instance_commit(id)
                    }
                }
            }
            if self.resize {
                self.resize_buffers();
                self.resize = false;
            }
            if self.rebuild {
                self.rebuild_linked_functions();
                self.rebuild_primitive_structures(cmd);
                self.rebuild_instance_structure(cmd);
                if self.rebuild_callback.is_some() {
                    let mut rebuild_callback = self.rebuild_callback.take().unwrap();
                    (rebuild_callback)(self);
                    self.rebuild_callback = Some(rebuild_callback);
                }
                self.rebuild = false;
            }
        }
        pub fn linked(&self) -> Option<&MTLLinkedFunctions> {
            self.linked.as_ref().map(|x| &**x)
        }
        pub fn functions(&self) -> Option<&[FunctionProtocol]> {
            self.functions.as_ref().map(|x| x.as_slice())
        }
        pub fn protocol(&self) -> Option<&AccelerationStructureProtocol> {
            self.structure.as_ref()
        }
        pub fn instance_data_buffer(&self) -> &BufferProtocol {
            &self.instance_indirect
        }
        pub fn per_instance_data_buffer(&self) -> &BufferProtocol {
            &self.per_instance_data
        }
        fn build_intersect_functions(
            names: Vec<String>,
            library: &LibraryProtocol,
        ) -> (Vec<FunctionProtocol>, Retained<MTLLinkedFunctions>) {
            let functions = names
                .into_iter()
                .map(|name| {
                    let intersect_desc = MTLIntersectionFunctionDescriptor::init(
                        MTLIntersectionFunctionDescriptor::alloc(),
                    );
                    intersect_desc.setName(Some(&NSString::from_str(&name)));
                    let constant_values =
                        MTLFunctionConstantValues::init(MTLFunctionConstantValues::alloc());
                    unsafe {
                        constant_values.setConstantValue_type_atIndex(
                            NonNull::from_ref(&0u32).cast(),
                            MTLDataType::UInt,
                            1,
                        );
                        constant_values.setConstantValue_type_atIndex(
                            NonNull::from_ref(&true).cast(),
                            MTLDataType::Bool,
                            1,
                        );
                    }
                    intersect_desc.setConstantValues(Some(&constant_values));
                    library
                        .newIntersectionFunctionWithDescriptor_error(&intersect_desc)
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let function_refs = functions.iter().map(|x| &**x).collect::<Vec<_>>();

            let linked = MTLLinkedFunctions::init(MTLLinkedFunctions::alloc());
            linked.setFunctions(Some(&NSArray::from_slice(&*function_refs)));
            (functions, linked)
        }

        fn resize_buffers(&mut self) {
            if self.bbox_used >= self.bbox_capacity {
                self.resize_bbox();
            }
            if self.instance_indirect_used >= self.instance_indirect_capacity {
                self.resize_instance();
            }
            if self.per_instance_data_used >= self.per_instance_data_capacity {
                self.resize_per_instance_data();
            }
        }

        fn resize_bbox(&mut self) {
            let new_capacity = (self.bbox_capacity * 2).max(self.bbox_used);
            let new_size = new_capacity * std::mem::size_of::<BoundingBox>();

            let new_buffer = self
                .device
                .newBufferWithLength_options(new_size, MTLResourceOptions::StorageModeShared)
                .expect("Failed to resize bbox buffer");

            // Copy existing data
            let old_ptr = self.bbox.contents().cast::<u8>().as_ptr();
            let new_ptr = new_buffer.contents().cast::<u8>().as_ptr();
            let copy_size = self.bbox_used * std::mem::size_of::<BoundingBox>();
            unsafe { std::ptr::copy_nonoverlapping(old_ptr, new_ptr, copy_size) };

            unsafe {
                let _: () = msg_send![&*self.residency, addAllocation: &*new_buffer];
            }
            self.bbox = new_buffer;
            self.bbox_capacity = new_capacity;

            // Mark all geometries as dirty
            for geometry in self.geometry.values_mut() {
                geometry.dirty = true;
            }
        }

        fn resize_instance(&mut self) {
            let new_capacity =
                (self.instance_indirect_capacity * 2).max(self.instance_indirect_used);
            let new_size = new_capacity
                * std::mem::size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>();

            let new_buffer = self
                .device
                .newBufferWithLength_options(new_size, MTLResourceOptions::StorageModeShared)
                .expect("Failed to resize instance buffer");

            unsafe {
                let _: () = msg_send![&*self.residency, addAllocation: &*new_buffer];
            }
            self.instance_indirect = new_buffer;
            self.instance_indirect_capacity = new_capacity;

            // Mark all instances as dirty
            for instance in self.instance.values_mut() {
                instance.dirty = true;
            }
        }

        fn resize_per_instance_data(&mut self) {
            //TODO
        }

        fn rebuild_linked_functions(&mut self) {
            let sorted_geometry = self
                .geometry
                .keys()
                .copied()
                .collect::<BTreeSet<Geometry>>();
            let mut sorted_geometry_names = sorted_geometry
                .into_iter()
                .map(|x| &self.geometry[&x])
                .map(|x| x.descriptor.func.name.clone())
                .collect::<Vec<_>>();
            sorted_geometry_names.dedup();

            self.geometry_intersector = Default::default();
            for (geometry, data) in &self.geometry {
                let mut idx = None;
                for (i, name) in sorted_geometry_names.iter().enumerate() {
                    if name == &data.descriptor.func.name {
                        idx = Some(i);
                        break;
                    }
                }
                self.geometry_intersector.insert(*geometry, idx.unwrap());
            }

            let (functions, linked) =
                Self::build_intersect_functions(sorted_geometry_names, &self.library);

            self.functions = Some(functions);
            self.linked = Some(linked);
        }

        fn rebuild_instance_structure(&mut self, cmd: &CommandBufferProtocol) {
            let instance_ptr = self
                .instance_indirect
                .contents()
                .cast::<MTLIndirectAccelerationStructureInstanceDescriptor>()
                .as_ptr();

            for (id, data) in self.instance.iter_mut().filter(|(_, data)| data.dirty) {
                let geometry = self
                    .geometry
                    .get(&data.descriptor.geometry)
                    .expect("instance references non existant geometry");

                let transform = Matrix::from_translation(data.descriptor.position)
                    * data.descriptor.rotation.to_matrix()
                    * Matrix::from_scale(data.descriptor.scale);
                let opaque = if geometry.descriptor.opaque {
                    MTLAccelerationStructureInstanceOptions::Opaque
                } else {
                    MTLAccelerationStructureInstanceOptions::NonOpaque
                };
                let mask = data.descriptor.mask;
                let primitive = geometry.primitive.as_ref().unwrap().gpuResourceID();

                unsafe {
                    *instance_ptr.add(data.index) =
                        MTLIndirectAccelerationStructureInstanceDescriptor {
                            transformationMatrix: transform.into(),
                            options: opaque,
                            mask,
                            intersectionFunctionTableOffset: 0,
                            userID: data.index as u32,
                            accelerationStructureID: primitive,
                        }
                }
            }

            let instance_count = self.instance.len();
            if instance_count == 0 {
                self.structure = None;
                return;
            }

            *unsafe { self.instance_count.contents().cast::<u32>().as_mut() } =
                instance_count as u32;

            let mut descriptor = MTL4IndirectInstanceAccelerationStructureDescriptor::init(
                MTL4IndirectInstanceAccelerationStructureDescriptor::alloc(),
            );

            unsafe {
                descriptor.setMaxInstanceCount(instance_count);
                descriptor.setInstanceDescriptorStride(std::mem::size_of::<
                    MTLIndirectAccelerationStructureInstanceDescriptor,
                >());
                descriptor.setInstanceCountBuffer(MTL4BufferRange {
                    bufferAddress: self.instance_count.gpuAddress(),
                    length: 32,
                });
                descriptor.setInstanceDescriptorBuffer(MTL4BufferRange {
                    bufferAddress: self.instance_indirect.gpuAddress(),
                    length: (instance_count
                        * std::mem::size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>())
                        as u64,
                });
                descriptor.setInstanceDescriptorType(
                    MTLAccelerationStructureInstanceDescriptorType::Indirect,
                );
            }
            unsafe {
                // CRITICAL: Ensure all buffers are in the residency set and committed
                let _: () = msg_send![&*self.residency, addAllocation: &*self.bbox];
                let _: () = msg_send![&*self.residency, addAllocation: &*self.instance_indirect];
                let _: () = msg_send![&*self.residency, addAllocation: &*self.instance_count];
                let _: () = msg_send![&*self.residency, addAllocation: &*self.per_instance_data];
                let _: () = msg_send![&*self.residency, addAllocation: &*self.scratch];

                self.residency.commit();
                self.residency.requestResidency();

                // Use the residency set AFTER committing and requesting residency
                cmd.useResidencySet(&self.residency);
            }
            let structure = unsafe { self.build_acceleration_structure(&cmd, &descriptor) };
            self.structure = Some(structure);
        }
        fn rebuild_primitive_structures(&mut self, cmd: &CommandBufferProtocol) {
            let mut geometry = mem::take(&mut self.geometry);

            for (geometry, mut data) in geometry
                .iter_mut()
                .filter(|(_, x)| x.dirty || x.primitive.is_none())
            {
                let intersection_function_table_offset = self.geometry_intersector[&geometry];

                let mut primitive_geometry_descriptor =
                    MTL4AccelerationStructureBoundingBoxGeometryDescriptor::init(
                        MTL4AccelerationStructureBoundingBoxGeometryDescriptor::alloc(),
                    );

                unsafe {
                    primitive_geometry_descriptor.setBoundingBoxCount(1);
                    primitive_geometry_descriptor
                        .setBoundingBoxStride(mem::size_of::<BoundingBox>());
                }
                let bbox_addr_base = self.bbox.gpuAddress();
                let bbox_addr_offset = (data.bbox_index * mem::size_of::<BoundingBox>()) as u64;
                let bbox_addr = bbox_addr_base + bbox_addr_offset;

                primitive_geometry_descriptor.setBoundingBoxBuffer(MTL4BufferRange {
                    bufferAddress: bbox_addr,
                    length: mem::size_of::<BoundingBox>() as u64,
                });
                unsafe {
                    primitive_geometry_descriptor
                        .setIntersectionFunctionTableOffset(intersection_function_table_offset);
                }
                primitive_geometry_descriptor.setOpaque(data.descriptor.opaque);

                let primitive_descriptor = MTL4PrimitiveAccelerationStructureDescriptor::init(
                    MTL4PrimitiveAccelerationStructureDescriptor::alloc(),
                );

                let geometry_array = NSArray::from_slice(&[&*primitive_geometry_descriptor]);
                unsafe {
                    let _: () =
                        msg_send![&*primitive_descriptor, setGeometryDescriptors:&*geometry_array];
                }

                unsafe {
                    // CRITICAL: Ensure all buffers are in the residency set and committed
                    let _: () = msg_send![&*self.residency, addAllocation: &*self.bbox];
                    let _: () =
                        msg_send![&*self.residency, addAllocation: &*self.instance_indirect];
                    let _: () = msg_send![&*self.residency, addAllocation: &*self.instance_count];
                    let _: () =
                        msg_send![&*self.residency, addAllocation: &*self.per_instance_data];
                    let _: () = msg_send![&*self.residency, addAllocation: &*self.scratch];

                    self.residency.commit();
                    self.residency.requestResidency();

                    // Use the residency set AFTER committing and requesting residency
                    cmd.useResidencySet(&self.residency);
                }
                let acceleration_structure =
                    unsafe { self.build_acceleration_structure(&cmd, &primitive_descriptor) };
                data.primitive = Some(acceleration_structure);
            }
            self.geometry = geometry;
        }

        fn add_geometry_commit(
            &mut self,
            cmd: &ProtocolObject<dyn MTL4CommandBuffer + 'static>,
            id: Geometry,
            descriptor: GeometryDescriptor,
        ) {
            if self.bbox_used >= self.bbox_capacity {
                self.resize = true;
            }

            let bbox_index = self.bbox_used;
            unsafe {
                *self
                    .bbox
                    .contents()
                    .cast::<BoundingBox>()
                    .as_ptr()
                    .offset(bbox_index as isize) = descriptor.bounding_box
            };

            self.geometry.insert(
                id,
                GeometryData {
                    primitive: None,
                    descriptor,
                    bbox_index,
                    dirty: true,
                },
            );
            self.bbox_used += 1;
        }

        fn remove_geometry_commit(
            &mut self,
            cmd: &ProtocolObject<dyn MTL4CommandBuffer + 'static>,
            id: Geometry,
        ) {
            self.geometry.remove(&id);
            self.instance
                .retain(|_, instance| instance.descriptor.geometry != id);
        }

        fn add_instance_commit(
            &mut self,
            cmd: &ProtocolObject<dyn MTL4CommandBuffer + 'static>,
            id: Instance,
            descriptor: InstanceDescriptor,
        ) {
            if self.instance_indirect_used >= self.instance_indirect_capacity {
                self.resize = true;
            }
            if self.per_instance_data_used >= self.per_instance_data_capacity {
                self.resize = true;
            }

            let index = self.instance_indirect_used;
            self.instance_indirect_used += 1;
            self.per_instance_data_used += 1;

            let transform = (Matrix::from_translation(descriptor.position)
                * descriptor.rotation.to_matrix()
                * Matrix::from_scale(descriptor.scale));
            let inverse_transform = transform.inverse().unwrap();

            let min_4d = Vector([
                descriptor.position[0],
                descriptor.position[1],
                descriptor.position[2],
                0.0,
            ]);
            let size_4d = Vector([
                descriptor.scale[0],
                descriptor.scale[1],
                descriptor.scale[2],
                0.0,
            ]);

            unsafe {
                *self
                    .per_instance_data
                    .contents()
                    .cast::<PerInstanceData>()
                    .as_ptr()
                    .offset(index as isize) = PerInstanceData {
                    transform,
                    inverse_transform,
                    min: min_4d,
                    size: size_4d,
                };
            }

            self.instance_to_offset.insert(id, index);
            self.instance.insert(
                id,
                InstanceData {
                    descriptor,
                    index,
                    dirty: true,
                },
            );
        }

        fn update_instance_commit(
            &mut self,
            cmd: &ProtocolObject<dyn MTL4CommandBuffer + 'static>,
            id: Instance,
            descriptor: InstanceDescriptor,
        ) {
            if let Some(instance) = self.instance.get_mut(&id) {
                instance.descriptor = descriptor;

                let transform = (Matrix::from_translation(descriptor.position)
                    * descriptor.rotation.to_matrix()
                    * Matrix::from_scale(descriptor.scale));
                let inverse_transform = transform.inverse().unwrap();
                let min_4d = Vector([
                    descriptor.position[0],
                    descriptor.position[1],
                    descriptor.position[2],
                    0.0,
                ]);
                let size_4d = Vector([
                    descriptor.scale[0],
                    descriptor.scale[1],
                    descriptor.scale[2],
                    0.0,
                ]);
                unsafe {
                    *self
                        .per_instance_data
                        .contents()
                        .cast::<PerInstanceData>()
                        .as_ptr()
                        .offset(instance.index as isize) = PerInstanceData {
                        transform,
                        inverse_transform,
                        min: min_4d,
                        size: size_4d,
                    };
                }
                instance.dirty = true;
            }
        }

        fn remove_instance_commit(&mut self, id: Instance) {
            self.instance.remove(&id);
            self.instance_to_offset.remove(&id);
        }
        pub unsafe fn build_acceleration_structure(
            &mut self,
            command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
            accel_desc: &MTL4AccelerationStructureDescriptor,
        ) -> Retained<ProtocolObject<dyn MTLAccelerationStructure>> {
            let accel_size = self
                .device
                .accelerationStructureSizesWithDescriptor(accel_desc);

            let accel_struct = self
                .device
                .newAccelerationStructureWithSize(accel_size.accelerationStructureSize)
                .unwrap();

            dbg!(accel_size.buildScratchBufferSize);

            let _: () = msg_send![&*self.residency, addAllocation: &*accel_struct];
            self.residency.commit();
            self.residency.requestResidency();

            command_buffer.useResidencySet(&self.residency);

            let encoder = command_buffer.computeCommandEncoder().unwrap();

            encoder.buildAccelerationStructure_descriptor_scratchBuffer(
                &*accel_struct,
                &*accel_desc,
                MTL4BufferRange {
                    bufferAddress: self.scratch.gpuAddress() + self.scratch_offset as u64,
                    length: accel_size.buildScratchBufferSize as u64,
                },
            );
            self.scratch_offset += accel_size.buildScratchBufferSize;

            encoder.endEncoding();

            accel_struct
        }
    }
}

pub type TextureProtocol = Retained<ProtocolObject<dyn MTLTexture + 'static>>;
pub type DeviceProtocol = Retained<ProtocolObject<dyn MTLDevice>>;
pub type ResidencyProtocol = Retained<ProtocolObject<dyn MTLResidencySet>>;
pub type CommandQueueProtocol = Retained<ProtocolObject<dyn MTL4CommandQueue>>;
pub type CommandBufferProtocol = Retained<ProtocolObject<dyn MTL4CommandBuffer>>;
pub type CommandAllocatorProtocol = Retained<ProtocolObject<dyn MTL4CommandAllocator>>;
pub type LibraryProtocol = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type EventProtocol = Retained<ProtocolObject<dyn MTLSharedEvent>>;
pub type ComputePipelineProtocol = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type ArgumentTableProtocol = Retained<ProtocolObject<dyn MTL4ArgumentTable>>;
pub type BufferProtocol = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type FunctionProtocol = Retained<ProtocolObject<dyn MTLFunction>>;
pub type IntersectionFunctionTableProtocol =
    Retained<ProtocolObject<dyn MTLIntersectionFunctionTable>>;
pub type RenderPipelineProtocol = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

pub struct Metal {
    frame: usize,
    frame_in_flight: usize,
    render_size: Size,
    render_texture: Vec<TextureProtocol>,
    next_texture: Box<dyn Fn() -> (TextureProtocol, Size)>,
    next_present: Option<Box<dyn Fn(&mut Metal)>>,
    device: DeviceProtocol,
    residency_set: ResidencyProtocol,
    command_queue: CommandQueueProtocol,
    command_allocator: Vec<CommandAllocatorProtocol>,
    library: LibraryProtocol,
    frame_event: EventProtocol,
    argument_table: Vec<ArgumentTableProtocol>,
    argument_table2: Vec<ArgumentTableProtocol>,
    camera_buffer: Vec<BufferProtocol>,
    palette_buffer: BufferProtocol,
    world_position_texture: Vec<TextureProtocol>,
    world_normal_texture: Vec<TextureProtocol>,
    shading: RenderPipelineProtocol,
    acceleration_structure: rtx::AccelerationStructure,
    raytrace: Option<ComputePipelineProtocol>,
    intersect_table: Option<IntersectionFunctionTableProtocol>,
    acoustics: Acoustics,
}

impl Metal {
    unsafe fn init(
        device: DeviceProtocol,
        next_texture: Box<dyn Fn() -> (TextureProtocol, Size)>,
        next_present: Box<dyn Fn(&mut Metal)>,
    ) -> Self {
        let mut engine = Engine::init().unwrap();

        // 1. Create the synth
        let synth = Arc::new(Synthesizer::new());

        // 2. Register it with the engine (Engine holds one Arc)
        engine.add_source(synth.clone());

        // 3. Sequencer Logic (Main Thread)
        // We use the local Arc 'synth' to control the sound.
        let env = Envelope {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.5,
            release: 0.2,
        };

        synth.note_on(
            Note::C.octave(5),
            Waveform::Sine,
            Envelope {
                attack: 0.1,
                decay: 0.5,
                sustain: 0.3,
                release: 0.2,
            },
        );
        std::thread::sleep_ms(10000);
        let frame_in_flight = 3;

        let next_present = Some(next_present);

        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            panic!("Metal 4 not supported.");
        }

        let command_queue: CommandQueueProtocol = msg_send![&device, newMTL4CommandQueue];
        let command_allocator = (0..frame_in_flight)
            .map(|_| {
                device
                    .newCommandAllocator()
                    .expect("failed to create command allocator")
            })
            .collect::<Vec<_>>();

        let source = include_str!("ui3d.metal");

        let source = NSString::from_str(source);

        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Failed to compile shader");

        let shading = {
            let func_name = NSString::from_str("upscale_vert");
            let vertex = library
                .newFunctionWithName(&func_name)
                .expect("Function not found");

            let func_name = NSString::from_str("shade_frag");
            let fragment = library
                .newFunctionWithName(&func_name)
                .expect("Function not found");

            let attachment = MTLRenderPipelineColorAttachmentDescriptor::init(
                MTLRenderPipelineColorAttachmentDescriptor::alloc(),
            );

            attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);

            let desc = MTLRenderPipelineDescriptor::init(MTLRenderPipelineDescriptor::alloc());
            desc.setVertexFunction(Some(&vertex));
            desc.setFragmentFunction(Some(&fragment));
            desc.colorAttachments()
                .setObject_atIndexedSubscript(Some(&attachment), 0);
            device
                .newRenderPipelineStateWithDescriptor_error(&desc)
                .expect("Failed to create pipeline")
        };
        let frame_event = device
            .newSharedEvent()
            .expect("Could not create shared event");
        let _: () = msg_send![&frame_event, setSignaledValue: 0u64];

        let camera_buffer = (0..frame_in_flight)
            .map(|_| {
                device
                    .newBufferWithLength_options(512, MTLResourceOptions::StorageModeShared)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut palette_data = vec![];
        const PALETTE: &str = include_str!("colors.palette");
        fn hex_to_rgb(hex_code: &str) -> Option<Vector<3, f32>> {
            // Remove the '#' prefix if present
            let hex_code = hex_code.strip_prefix('#').unwrap_or(hex_code);

            // Ensure the string has the correct length (6 characters for RRGGBB)
            if hex_code.len() != 6 {
                return None;
            }

            // Parse each two-character segment into a u8
            let r = u8::from_str_radix(&hex_code[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex_code[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex_code[4..6], 16).ok()?;

            Some(Vector([r as f32 / 255., g as f32 / 255., b as f32 / 255.]))
        }

        pub fn rgb_to_hsv(this: Vector<3, f32>) -> Vector<3, f32> {
            let Vector([r, g, b]) = this;
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            // 1. Value
            let v = max;

            // 2. Saturation
            let s = if max == 0.0 { 0.0 } else { delta / max };

            // 3. Hue
            let h = if delta == 0.0 {
                0.0 // Grayscale
            } else {
                let mut hue = if max == r {
                    ((g - b) / delta) % 6.0 // Red sector
                } else if max == g {
                    ((b - r) / delta) + 2.0 // Green sector
                } else {
                    ((r - g) / delta) + 4.0 // Blue sector
                };

                // Normalize to 0-1 range
                hue = hue / 6.0;

                // Wrap negative values
                if hue < 0.0 {
                    hue += 1.0;
                }

                hue
            };

            Vector([h, s, v])
        }
        for hex in PALETTE.lines() {
            palette_data.push(dbg!(rgb_to_hsv(hex_to_rgb(hex).unwrap())));
        }
        let mut palette_data_slice = unsafe {
            slice::from_raw_parts_mut(
                palette_data.as_mut_ptr().cast::<u8>(),
                mem::size_of::<Vector<3, f32>>() * palette_data.len(),
            )
        };
        let palette_buffer = device
            .newBufferWithBytes_length_options(
                NonNull::new_unchecked(palette_data_slice.as_mut_ptr()).cast(),
                palette_data_slice.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();

        let render_size = Size::new(1280.0, 720.0);
        let texture_descriptor = MTLTextureDescriptor::new();
        texture_descriptor.setTextureType(MTLTextureType::Type2D);
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setWidth(render_size.width as usize);
        texture_descriptor.setHeight(render_size.height as usize);
        texture_descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Tracked);
        texture_descriptor.setUsage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        let render_texture = (0..frame_in_flight)
            .map(|_| {
                device
                    .newTextureWithDescriptor(&texture_descriptor)
                    .expect("Failed to create render texture")
            })
            .collect::<Vec<_>>();

        texture_descriptor.setPixelFormat(MTLPixelFormat::RGBA32Float);
        let world_position_texture = (0..frame_in_flight)
            .map(|_| {
                device
                    .newTextureWithDescriptor(&texture_descriptor)
                    .expect("Failed to create render texture")
            })
            .collect::<Vec<_>>();

        let world_normal_texture = (0..frame_in_flight)
            .map(|_| {
                device
                    .newTextureWithDescriptor(&texture_descriptor)
                    .expect("Failed to create render texture")
            })
            .collect::<Vec<_>>();

        let residency_desc = MTLResidencySetDescriptor::init(MTLResidencySetDescriptor::alloc());

        let residency_set = device
            .newResidencySetWithDescriptor_error(&residency_desc)
            .unwrap();

        for i in 0..frame_in_flight {
            let _: () = msg_send![&*residency_set, addAllocation: &*render_texture[i]];
            let _: () = msg_send![&*residency_set, addAllocation: &*world_position_texture[i]];
            let _: () = msg_send![&*residency_set, addAllocation: &*world_normal_texture[i]];
            let _: () = msg_send![&*residency_set, addAllocation: &*camera_buffer[i]];
        }

        let argument_table_descriptor =
            MTL4ArgumentTableDescriptor::init(MTL4ArgumentTableDescriptor::alloc());
        argument_table_descriptor.setMaxTextureBindCount(4);
        argument_table_descriptor.setMaxBufferBindCount(4);

        let argument_table2 = (0..frame_in_flight)
            .map(|_| {
                device
                    .newArgumentTableWithDescriptor_error(&argument_table_descriptor)
                    .expect("failed to make argument table")
            })
            .collect();
        let argument_table = (0..frame_in_flight)
            .map(|_| {
                device
                    .newArgumentTableWithDescriptor_error(&argument_table_descriptor)
                    .expect("failed to make argument table")
            })
            .collect();

        residency_set.commit();
        residency_set.requestResidency();

        let mut acceleration_structure =
            rtx::AccelerationStructure::new(&device, &residency_set, &library);

        use memvox::Volume;
        let mut vol = memvox::MemVol::<usize>::with_capacity(0, Vector([8usize, 8, 8]));

        dbg!(vol.capacity());
        let perlin = &noise::Perlin::new(32);
        for x in 0..8 {
            for y in 0..8 {
                use noise::NoiseFn;
                let top = NoiseFn::<f64, 3>::get(&perlin, [x as f64 * 0.05, y as f64 * 0.05, 0.])
                    as f32
                    * 12.
                    + 4.;

                for z in 0..top as usize {
                    vol.set(Vector([x, y, z]), 1);
                }
            }
        }

        let mut bounding_boxes = memvox::BoundingBox::derive(&vol, |x| *x == 1);
        let sphere = acceleration_structure.add_geometry(GeometryDescriptor {
            bounding_box: BoundingBox {
                min: array::from_fn(|_| EPSILON),
                max: array::from_fn(|_| 1.0 - EPSILON),
            },
            func: rtx::IntersectionFunction {
                name: "voxel_intersect".into(),
            },
            opaque: false,
        });
        for memvox::BoundingBox { size, min, .. } in bounding_boxes.into_iter() {
            dbg!(min, size);
            acceleration_structure.add_instance(InstanceDescriptor {
                geometry: sphere,
                rotation: Quaternion::IDENTITY,
                position: Vector([min[0] as f32, min[1] as f32, min[2] as f32]),
                scale: Vector([size[0] as f32, size[1] as f32, size[2] as f32]),
                mask: 1,
                user_id: 0,
            });
        }

        let plane = acceleration_structure.add_geometry(GeometryDescriptor {
            bounding_box: BoundingBox {
                min: [-1000.0, -1000.0, -0.1],
                max: [1000.0, 1000.0, 0.0],
            },
            func: rtx::IntersectionFunction {
                name: "grid_intersect".into(),
            },
            opaque: false,
        });
        use math::One;
        acceleration_structure.add_instance(InstanceDescriptor {
            geometry: plane,
            rotation: Quaternion::IDENTITY,
            position: Vector::ZERO,
            scale: Vector::<3, f32>::ONE,
            mask: 1,
            user_id: 0,
        });

        acceleration_structure.set_rebuild_callback(|accel| REBUILT = true);

        let acoustics = unsafe { mem::zeroed() }; //::init(&device, &library);

        Self {
            frame: 0,
            frame_in_flight,
            camera_buffer,
            render_size,
            render_texture,
            next_texture,
            next_present,
            device,
            residency_set,
            command_queue,
            command_allocator,
            library,
            frame_event,
            argument_table,
            argument_table2,
            world_position_texture,
            world_normal_texture,
            shading,
            acceleration_structure,
            raytrace: None,
            intersect_table: None,
            palette_buffer,
            acoustics,
        }
    }

    unsafe fn render(&mut self, camera: CameraData) {
        if self.frame >= self.frame_in_flight {
            let frame_to_wait = self.frame - self.frame_in_flight;
            let done: bool = msg_send![&self.frame_event, waitUntilSignaledValue:frame_to_wait, timeoutMS:10usize];
        }

        let frame_index = self.frame % self.frame_in_flight;
        self.frame += 1;

        let (id, size) = (self.next_texture)();

        let command_buffer = self.device.newCommandBuffer().unwrap();
        self.command_allocator[frame_index].reset();

        let residency_set = &self.residency_set;

        residency_set.commit();
        residency_set.requestResidency();

        command_buffer.beginCommandBufferWithAllocator(&self.command_allocator[frame_index]);

        command_buffer.useResidencySet(&self.residency_set);
        self.acceleration_structure.update(&command_buffer);

        // Bind G-Buffer textures for the compute pass
        self.argument_table[frame_index]
            .setTexture_atIndex(self.world_position_texture[frame_index].gpuResourceID(), 0);
        self.argument_table[frame_index]
            .setTexture_atIndex(self.world_normal_texture[frame_index].gpuResourceID(), 1);

        // Bind buffers for the compute pass
        self.argument_table[frame_index]
            .setAddress_atIndex(self.camera_buffer[frame_index].gpuAddress(), 0);
        let per_instance_buffer = self.acceleration_structure.per_instance_data_buffer();
        self.argument_table[frame_index].setAddress_atIndex(per_instance_buffer.gpuAddress(), 3);
        self.argument_table[frame_index].setAddress_atIndex(
            self.acceleration_structure
                .instance_data_buffer()
                .gpuAddress(),
            4,
        );

        if let Some(protocol) = self.acceleration_structure.protocol() {
            self.argument_table[frame_index].setResource_atBufferIndex(protocol.gpuResourceID(), 1);
        }
        if REBUILT && let Some(functions) = self.acceleration_structure.functions() {
            let func_name = NSString::from_str("raytrace");
            let rtx = self
                .library
                .newFunctionWithName(&func_name)
                .expect("Function not found");
            let mut descriptor =
                MTLComputePipelineDescriptor::init(MTLComputePipelineDescriptor::alloc());
            descriptor.setComputeFunction(Some(&*rtx));
            descriptor.setLinkedFunctions(Some(&*self.acceleration_structure.linked().unwrap()));
            let raytrace = self
                .device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &*descriptor,
                    MTLPipelineOption::empty(),
                    None,
                )
                .unwrap();

            let mut descriptor = MTLIntersectionFunctionTableDescriptor::new();
            descriptor.setFunctionCount(functions.len());

            let intersect_table = raytrace
                .newIntersectionFunctionTableWithDescriptor(&descriptor)
                .unwrap();
            for (index, function) in functions.iter().enumerate() {
                let handle = raytrace.functionHandleWithFunction(function).unwrap();
                intersect_table.setFunction_atIndex(Some(&*handle), index);
            }

            self.intersect_table = Some(intersect_table);

            self.raytrace = Some(raytrace);

            REBUILT = false;
        }
        if let Some(intersect_table) = self.intersect_table.as_ref() {
            self.argument_table[frame_index]
                .setResource_atBufferIndex(intersect_table.gpuResourceID(), 2);
            let _: () = msg_send![&*self.residency_set, addAllocation: &**intersect_table];
        }

        *self.camera_buffer[frame_index]
            .contents()
            .cast::<CameraData>()
            .as_mut() = camera;

        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("failed to make compute encoder");

        encoder.setArgumentTable(Some(&self.argument_table[frame_index]));
        let threadgroups_per_grid = MTLSize {
            width: self.render_size.width as usize / 16 + 1,
            height: self.render_size.height as usize / 16 + 1,
            depth: 1,
        };
        let threads_per_threadgroup = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        if let Some(raytrace) = &self.raytrace {
            encoder.setComputePipelineState(&raytrace);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                threadgroups_per_grid,
                threads_per_threadgroup,
            );
        }
        let probes = self.acoustics.probe(&self.device, &command_buffer);
        encoder.barrierAfterStages_beforeQueueStages_visibilityOptions(
            MTLStages::All,
            MTLStages::Fragment,
            MTL4VisibilityOptions::Device,
        );

        encoder.endEncoding();

        self.argument_table2[frame_index]
            .setTexture_atIndex(self.world_position_texture[frame_index].gpuResourceID(), 2);
        self.argument_table2[frame_index]
            .setTexture_atIndex(self.world_normal_texture[frame_index].gpuResourceID(), 3);
        self.argument_table2[frame_index].setAddress_atIndex(self.palette_buffer.gpuAddress(), 5);

        let attachment = MTLRenderPassColorAttachmentDescriptor::init(
            MTLRenderPassColorAttachmentDescriptor::alloc(),
        );

        attachment.setTexture(Some(&id));

        let render_descriptor = MTL4RenderPassDescriptor::init(MTL4RenderPassDescriptor::alloc());
        render_descriptor.setRenderTargetWidth(size.width as _);

        render_descriptor.setRenderTargetHeight(size.height as _);
        render_descriptor
            .colorAttachments()
            .setObject_atIndexedSubscript(Some(&attachment), 0);

        let render_encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&render_descriptor)
            .unwrap();

        render_encoder.setArgumentTable_atStages(
            &self.argument_table2[frame_index],
            MTLRenderStages::Fragment,
        );
        render_encoder.setRenderPipelineState(&self.shading);
        render_encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: size.width,
            height: size.height,
            znear: 0.0,
            zfar: 1.0,
        });

        render_encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);

        render_encoder.endEncoding();

        command_buffer.endCommandBuffer();

        LATEST_CMDS = Some(command_buffer);
        let mut present = self.next_present.take();
        (present.as_mut().unwrap())(self);
        self.next_present = present;
        let _: () =
            msg_send![&*self.command_queue, signalEvent:&*self.frame_event, value:self.frame];
        // Signal with the actual frame number, not the index
        println!("presenting!");
    }
}

pub struct Darwin {
    metal: RefCell<Metal>,
    window: Rc<RefCell<Window>>,
    camera_vel: RefCell<Vector<2, f32>>,
    camera_pos: RefCell<Vector<2, f32>>,
    angle_vel: RefCell<f32>,
    angle: RefCell<f32>,
    zoom_vel: RefCell<f32>,
    zoom: RefCell<f32>,
}
impl Darwin {
    pub unsafe fn init(data: *const ()) -> Self {
        let view_controller = data.cast::<UIViewController>().as_ref().unwrap();
        let window = Window::new(view_controller);
        let window = Rc::new(RefCell::new(window));
        let draw = window.clone();
        let present = window.clone();
        println!("Preparing to init metal");
        let metal = Metal::init(
            window.borrow_mut().device(),
            Box::new(move || draw.borrow_mut().next()),
            Box::new(move |metal| present.borrow_mut().draw(metal)),
        );
        let metal = RefCell::new(metal);
        Self {
            metal,
            window,
            camera_vel: RefCell::new(Vector::ZERO),
            camera_pos: RefCell::new(Vector([0.0, 0.0])),
            angle: (PI / 4.0).into(),
            zoom: 0.5.into(),
            angle_vel: 0.0.into(),
            zoom_vel: 0.0.into(),
        }
    }
    pub fn camera_data(&self) -> CameraData {
        let screen_size = self.metal.borrow().render_size;
        let native_scale = self.window.borrow().native_scale();
        let camera_pos = self.camera_pos.borrow();
        let projection = Projection {
            fov: PI / 2.0,
            aspect: 1.0,
            near: 0.1,
        };
        let center = Vector([camera_pos[0], camera_pos[1], 0.0]); // Looking at z=0
        let distance = (self.zoom.borrow().powf(2.0) * 100.0 + 30.0).clamp(30.0, 130.0);
        let base_direction = Vector([0.0, -1.0, 1.0 + 2.0 * *self.zoom.borrow()]).normalize(); // Start at 45° angle

        let up = Vector::<3, f32>::Z;
        let rotation = Quaternion::from_axis_angle(up, *self.angle.borrow());

        let rotated_direction = rotation * base_direction;
        let mut pos = center + rotated_direction * distance;
        dbg!(center, pos, rotated_direction, distance);
        let mut transform =
            Quaternion::look_rotation(rotated_direction, Vector::<3, f32>::Z).to_matrix();
        for i in 0..3 {
            transform[(i, 3)] = pos[i];
        }
        dbg!(center);
        CameraData {
            projection: dbg!(projection.to_matrix()),
            projection_inverse: dbg!(projection.to_matrix().inverse().unwrap()),
            view: transform.inverse().unwrap(),
            transform,
            resolution: Vector([
                screen_size.width as f32,
                screen_size.height as f32,
                native_scale as f32,
                0.0,
            ]),
            info: Vector([self.metal.borrow().frame as u32, 0, 0, 0]),
        }
    }
    fn pan_screen_in_world(
        &self,
        location: Vector<2, f32>,
        velocity: Vector<2, f32>,
    ) -> Option<Vector<3, f32>> {
        let camera = self.camera_data();
        let screen_size = self.window.borrow().screen_size();

        let before =
            dbg!(camera.unproject(location, screen_size, self.window.borrow().native_scale()));

        let after = dbg!(camera.unproject(
            location + velocity,
            screen_size,
            self.window.borrow().native_scale()
        ));

        let ground = Plane::Z;
        dbg!(Some(ground.intersect(after)? - ground.intersect(before)?))
    }
    pub fn draw(&self) {
        let input = self.window.borrow_mut().take_input();
        for input in input {
            match input {
                Input::Move(velocity) => {
                    let up = Vector::<3, f32>::Z;
                    let rotation = Quaternion::from_axis_angle(up, *self.angle.borrow());
                    let movement = (rotation * velocity.expand(0.)) * 10.;
                    *self.camera_vel.borrow_mut() = Vector([movement[0], movement[1]])
                }
                Input::Pan(screen_position, screen_velocity) => {
                    dbg!(screen_position, screen_velocity);
                    let vel = self
                        .pan_screen_in_world(screen_position, screen_velocity)
                        .unwrap_or(Vector::ZERO);
                    dbg!(vel);
                    *self.camera_vel.borrow_mut() = Vector([vel[0], vel[1]]);
                }
                Input::Rotate(velocity) => {
                    *self.angle_vel.borrow_mut() = velocity;
                }
                Input::Zoom(zoom) => {
                    *self.zoom_vel.borrow_mut() = zoom;
                }
            }
        }
        *self.angle.borrow_mut() += *self.angle_vel.borrow() / 60.0;
        *self.angle_vel.borrow_mut() *= 0.8;
        *self.camera_pos.borrow_mut() += *self.camera_vel.borrow() / 60.0;
        *self.camera_vel.borrow_mut() *= 0.8;
        *self.zoom.borrow_mut() += *self.zoom_vel.borrow_mut() / 60.0;
        let current_zoom = *self.zoom.borrow();
        *self.zoom.borrow_mut() = dbg!(current_zoom.clamp(0.0, 1.0));
        *self.zoom_vel.borrow_mut() *= 0.8;
        let camera_data = self.camera_data();
        unsafe { self.metal.borrow_mut().render(camera_data) }
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
