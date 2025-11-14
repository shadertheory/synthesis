use core::slice;
use std::f32::consts::PI;
use std::rc::Rc;
use std::time::Duration;
use std::{cell::RefCell, ptr::NonNull};
use std::{mem, ptr, thread};

use objc2::{rc::*, runtime::*, *};
use objc2_core_graphics::*;
use objc2_foundation::*;
use objc2_metal::*;
use objc2_metal_kit::*;
use objc2_quartz_core::*;
use objc2_ui_kit::*;

use crate::math::*;

use crate::{CameraData, Input, Matrix, Projection, Quaternion, Vector};

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

    pub fn size(&self) -> crate::Vector<2, f32> {
        let bounds: NSRect = unsafe { msg_send![&*self.screen, bounds] };
        crate::Vector([bounds.size.width as f32, bounds.size.height as f32])
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

pub static mut LATEST_CMDS: Option<CommandBufferProtocol> = None;

pub struct Window {
    surface: Surface,
    pan: Retained<UIPanGestureRecognizer>,
    pinch: Retained<UIPinchGestureRecognizer>,
    rotate: Retained<UIRotationGestureRecognizer>,
    inputs: Vec<Input>,
}
impl Window {
    pub fn screen_size(&self) -> crate::Vector<2, f32> {
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
    pub fn poll_input(&mut self) {
        unsafe {
            self.update_pan();
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
impl From<NSPoint> for crate::Vector<2, f32> {
    fn from(point: NSPoint) -> Self {
        Self([point.x as f32, point.y as f32])
    }
}
mod rtx {
    use std::{collections::HashMap, hash::Hash, mem, ptr::NonNull};

    use crate::{
        Quaternion, Vector,
        darwin::{
            BufferProtocol, CommandBufferProtocol, DeviceProtocol, LibraryProtocol,
            ResidencyProtocol,
        },
    };

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

    pub struct IntersectionFunction {
        pub name: String,
    }

    pub struct GeometryDescriptor {
        pub bounding_box: BoundingBox,
        pub func: IntersectionFunction,
        pub opaque: bool,
    }

    pub struct InstanceDescriptor {
        pub geometry: Geometry,
        pub rotation: Quaternion<f32>,
        pub position: Vector<3, f32>,
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
        intersection_func_index: Option<usize>,
        dirty: bool,
    }
    pub struct InstanceData {
        descriptor: InstanceDescriptor,
        instance_index: usize,
        dirty: bool,
    }
    pub struct AccelerationStructure {
        device: DeviceProtocol,
        residency: ResidencyProtocol,

        geometry: HashMap<Geometry, GeometryData>,
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

        instance_count: BufferProtocol,

        library: LibraryProtocol,
        linked: Retained<MTLLinkedFunctions>,

        scratch: BufferProtocol,
        scratch_size: usize,

        instance_accel: Option<AccelerationStructureProtocol>,

        pending: Vec<PendingOp>,

        rebuild: bool,
        resize: bool,
    }

    impl AccelerationStructure {
        const MAX_FUNCTIONS: usize = 16;
        pub fn new(
            device: DeviceProtocol,
            residency: ResidencyProtocol,
            library: LibraryProtocol,
        ) -> Self {
            Self::with_capacity(device, residency, library, 256, 4096, 16 * 1024 * 1024)
        }
        pub fn with_capacity(
            device: DeviceProtocol,
            residency: ResidencyProtocol,
            library: LibraryProtocol,
            geometry_capacity: usize,
            instance_capacity: usize,
            scratch_size: usize,
        ) -> Self {
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

            let linked = Self::link_intersect_functions(vec![], &library);

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
                instance_indirect_capacity: geometry_capacity,
                instance_indirect_used: 0,
                instance_count,
                scratch,
                scratch_size,
                library,
                linked,
                instance_accel: None,
                pending: vec![],
                rebuild: false,
                resize: false,
            }
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
            let id = Instance(self.next_geometry);
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
                self.rebuild_structures();
                self.rebuild = false;
            }
        }
        fn link_intersect_functions(
            names: Vec<String>,
            library: &LibraryProtocol,
        ) -> Retained<MTLLinkedFunctions> {
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
            linked
        }

        fn resize_buffers(&mut self) {}

        fn rebuild_structures(&mut self) {}

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
                    intersection_func_index: None,
                    dirty: true,
                },
            );
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

            let instance_index = self.instance_indirect_used;
            self.instance_indirect_used += 1;

            self.instance_to_offset.insert(id, instance_index);
            self.instance.insert(
                id,
                InstanceData {
                    descriptor,
                    instance_index,
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
            }
        }

        fn remove_instance_commit(&mut self, id: Instance) {
            self.instance.remove(&id);
            self.instance_to_offset.remove(&id);
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
    ui_raytrace: ComputePipelineProtocol,
    argument_table: Vec<ArgumentTableProtocol>,
    argument_table2: Vec<ArgumentTableProtocol>,
    camera_buffer: Vec<BufferProtocol>,
    interesection_table: IntersectionFunctionTableProtocol,
    upscale: RenderPipelineProtocol,
    raytrace: ComputePipelineProtocol,
}

impl Metal {
    unsafe fn init(
        device: DeviceProtocol,
        next_texture: Box<dyn Fn() -> (TextureProtocol, Size)>,
        next_present: Box<dyn Fn(&mut Metal)>,
    ) -> Self {
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

        let ui_raytrace = {
            let func_name = NSString::from_str("reference_grid");
            let function = library
                .newFunctionWithName(&func_name)
                .expect("Function not found");
            device
                .newComputePipelineStateWithFunction_error(&function)
                .expect("Failed to create pipeline")
        };
        let upscale = {
            let func_name = NSString::from_str("upscale_vert");
            let vertex = library
                .newFunctionWithName(&func_name)
                .expect("Function not found");

            let func_name = NSString::from_str("upscale_frag");
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
        let raytrace = {
            let func_name = NSString::from_str("raytrace");
            let rtx = library
                .newFunctionWithName(&func_name)
                .expect("Function not found");
            let intersect_desc =
                MTLIntersectionFunctionDescriptor::init(MTLIntersectionFunctionDescriptor::alloc());
            intersect_desc.setName(Some(&NSString::from_str("sphere_intersect")));
            let constant_values =
                MTLFunctionConstantValues::init(MTLFunctionConstantValues::alloc());
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
            intersect_desc.setConstantValues(Some(&constant_values));
            let intersect = library
                .newIntersectionFunctionWithDescriptor_error(&intersect_desc)
                .unwrap();
            let linked = MTLLinkedFunctions::init(MTLLinkedFunctions::alloc());
            linked.setFunctions(Some(&NSArray::from_slice(&[&*intersect])));
            let mut desc =
                MTLComputePipelineDescriptor::init(MTLComputePipelineDescriptor::alloc());
            desc.setComputeFunction(Some(&*rtx));
            desc.setLinkedFunctions(Some(&*linked));
            device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &*desc,
                    MTLPipelineOption::empty(),
                    None,
                )
                .unwrap()
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
        let residency_desc = MTLResidencySetDescriptor::init(MTLResidencySetDescriptor::alloc());

        let residency_set = device
            .newResidencySetWithDescriptor_error(&residency_desc)
            .unwrap();

        for i in 0..frame_in_flight {
            let _: () = msg_send![&*residency_set, addAllocation: &*render_texture[i]];
            let _: () = msg_send![&*residency_set, addAllocation: &*camera_buffer[i]];
        }

        let intersection_function_table_desc = MTLIntersectionFunctionTableDescriptor::init(
            MTLIntersectionFunctionTableDescriptor::alloc(),
        );
        intersection_function_table_desc.setFunctionCount(1);

        // Create intersection function table from the PIPELINE STATE
        let intersection_function_table = raytrace
            .newIntersectionFunctionTableWithDescriptor(&intersection_function_table_desc)
            .expect("deez");

        // Get the intersection function handle (same as before)
        let intersect_name = NSString::from_str("sphere_intersect");
        let intersect_function_handle = raytrace
            .functionHandleWithFunction(&library.newFunctionWithName(&intersect_name).unwrap())
            .expect("Failed to get function handle");

        // Set it in the intersection function table at index 0
        intersection_function_table.setFunction_atIndex(Some(&intersect_function_handle), 0);

        // Add to residency set
        let _: () = msg_send![&*residency_set, addAllocation: &*intersection_function_table];

        // Get the intersection function handle
        let intersect_name = NSString::from_str("sphere_intersect");
        let intersect_function_handle = raytrace
            .functionHandleWithFunction(&library.newFunctionWithName(&intersect_name).unwrap())
            .expect("Failed to get function handle");

        residency_set.commit();
        residency_set.requestResidency();

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
            ui_raytrace,
            upscale,
            raytrace,
            interesection_table: intersection_function_table,
        }
    }

    pub unsafe fn create_accel(
        device: &ProtocolObject<dyn MTLDevice>,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        accel_desc: &MTL4AccelerationStructureDescriptor,
        residency_set: &ResidencyProtocol,
    ) -> Retained<ProtocolObject<dyn MTLAccelerationStructure>> {
        let accel_size = device.accelerationStructureSizesWithDescriptor(accel_desc);

        let accel_struct = device
            .newAccelerationStructureWithSize(accel_size.accelerationStructureSize)
            .unwrap();

        dbg!(accel_size.buildScratchBufferSize);

        let accel_scratch = device
            .newBufferWithLength_options(
                accel_size.buildScratchBufferSize,
                MTLResourceOptions::StorageModePrivate,
            )
            .unwrap();
        let accel_scratch = Box::leak(Box::new(accel_scratch));

        let _: () = msg_send![&*residency_set, addAllocation: &*accel_struct];
        let _: () = msg_send![&*residency_set, addAllocation: &**accel_scratch];
        residency_set.commit();
        residency_set.requestResidency();

        command_buffer.useResidencySet(residency_set);

        let encoder = command_buffer.computeCommandEncoder().unwrap();

        encoder.buildAccelerationStructure_descriptor_scratchBuffer(
            &*accel_struct,
            &*accel_desc,
            MTL4BufferRange {
                bufferAddress: accel_scratch.gpuAddress(),
                length: accel_size.buildScratchBufferSize as u64,
            },
        );

        encoder.endEncoding();

        accel_struct
    }
    unsafe fn render(&mut self, camera: crate::CameraData) {
        let instance_buffer = self
            .device
            .newBufferWithLength_options(
                std::mem::size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();

        let instance_count_buffer = self
            .device
            .newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared)
            .unwrap();

        *instance_count_buffer.contents().cast::<u32>().as_mut() = 1;
        let accel_desc = MTL4PrimitiveAccelerationStructureDescriptor::init(
            MTL4PrimitiveAccelerationStructureDescriptor::alloc(),
        );
        let bounding_box_buffer = self
            .device
            .newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let _: () = msg_send![&*self.residency_set, addAllocation: &*bounding_box_buffer];
        #[repr(C)]
        struct BoundingBox {
            min: [f32; 3],
            max: [f32; 3],
        }

        let bbox = BoundingBox {
            min: [-0.5, -0.5, -0.5],
            max: [0.5, 0.5, 0.5],
        };
        *bounding_box_buffer
            .contents()
            .cast::<BoundingBox>()
            .as_mut() = bbox;

        let geometry = MTL4AccelerationStructureBoundingBoxGeometryDescriptor::init(
            MTL4AccelerationStructureBoundingBoxGeometryDescriptor::alloc(),
        );
        geometry.setBoundingBoxCount(1);
        geometry.setBoundingBoxStride(24);
        geometry.setBoundingBoxBuffer(MTL4BufferRange {
            bufferAddress: bounding_box_buffer.gpuAddress(),
            length: 32,
        });
        geometry.setIntersectionFunctionTableOffset(0);
        geometry.setOpaque(false);
        let geometry_array = NSArray::from_slice(&[&*geometry]);
        let _: () = msg_send![&*accel_desc, setGeometryDescriptors:&*geometry_array];

        let mut instance_desc = MTL4IndirectInstanceAccelerationStructureDescriptor::init(
            MTL4IndirectInstanceAccelerationStructureDescriptor::alloc(),
        );

        instance_desc.setMaxInstanceCount(1);
        instance_desc.setInstanceDescriptorStride(mem::size_of::<
            MTLIndirectAccelerationStructureInstanceDescriptor,
        >());
        instance_desc.setInstanceCountBuffer(MTL4BufferRange {
            bufferAddress: instance_count_buffer.gpuAddress(),
            length: 32,
        });
        instance_desc.setInstanceDescriptorBuffer(MTL4BufferRange {
            bufferAddress: instance_buffer.gpuAddress(),
            length: mem::size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>() as _,
        });
        instance_desc
            .setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Indirect);

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
        let _: () = msg_send![&*residency_set, addAllocation: &*instance_count_buffer];
        let _: () = msg_send![&*residency_set, addAllocation: &*instance_buffer];

        residency_set.commit();
        residency_set.requestResidency();

        command_buffer.beginCommandBufferWithAllocator(&self.command_allocator[frame_index]);

        let prim_accel = Self::create_accel(
            &self.device,
            &command_buffer,
            &accel_desc,
            &self.residency_set,
        );
        *instance_buffer
            .contents()
            .cast::<MTLIndirectAccelerationStructureInstanceDescriptor>()
            .as_mut() = MTLIndirectAccelerationStructureInstanceDescriptor {
            transformationMatrix: MTLPackedFloat4x3 {
                columns: [
                    MTLPackedFloat3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    MTLPackedFloat3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    MTLPackedFloat3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    },
                    MTLPackedFloat3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    }, // translation
                ],
            },
            options: MTLAccelerationStructureInstanceOptions::NonOpaque,
            mask: 1,
            intersectionFunctionTableOffset: 0,
            userID: 0,
            accelerationStructureID: prim_accel.gpuResourceID(),
        };
        let _: () = msg_send![&*residency_set, addAllocation: &*prim_accel];
        residency_set.commit();
        residency_set.requestResidency();
        let instance_accel = Self::create_accel(
            &self.device,
            &command_buffer,
            &instance_desc,
            &self.residency_set,
        );
        let _: () = msg_send![&*residency_set, addAllocation: &*instance_accel];
        residency_set.commit();
        residency_set.requestResidency();
        self.argument_table[frame_index]
            .setTexture_atIndex(self.render_texture[frame_index].gpuResourceID(), 0);
        self.argument_table[frame_index]
            .setAddress_atIndex(self.camera_buffer[frame_index].gpuAddress(), 0);
        self.argument_table[frame_index]
            .setResource_atBufferIndex(instance_accel.gpuResourceID(), 1);
        self.argument_table[frame_index]
            .setResource_atBufferIndex(self.interesection_table.gpuResourceID(), 2);

        command_buffer.useResidencySet(&self.residency_set);

        *self.camera_buffer[frame_index]
            .contents()
            .cast::<CameraData>()
            .as_mut() = camera;

        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("failed to make compute encoder");

        encoder.setArgumentTable(Some(&self.argument_table[frame_index]));
        encoder.setComputePipelineState(&self.ui_raytrace);
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
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        encoder.setComputePipelineState(&self.raytrace);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        encoder.barrierAfterStages_beforeQueueStages_visibilityOptions(
            MTLStages::All,
            MTLStages::Fragment,
            MTL4VisibilityOptions::Device,
        );

        encoder.endEncoding();
        self.argument_table2[frame_index]
            .setTexture_atIndex(self.render_texture[frame_index].gpuResourceID(), 0);

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
        render_encoder.setRenderPipelineState(&self.upscale);
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
    camera_vel: RefCell<crate::Vector<2, f32>>,
    camera_pos: RefCell<crate::Vector<2, f32>>,
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
            camera_vel: RefCell::new(crate::Vector::ZERO),
            camera_pos: RefCell::new(crate::Vector([0.0, 0.0])),
            angle: (PI / 4.0).into(),
            zoom: 0.5.into(),
            zoom_vel: 0.0.into(),
        }
    }
    pub fn camera_data(&self) -> crate::CameraData {
        use crate::CameraData;
        let screen_size = self.metal.borrow().render_size;
        let native_scale = self.window.borrow().native_scale();
        let camera_pos = self.camera_pos.borrow();
        let projection = Projection {
            fov: PI / 2.0,
            aspect: 1.0,
            near: 0.1,
        };
        let center = crate::Vector([camera_pos[0], camera_pos[1], 0.0]); // Looking at z=0
        let distance = (self.zoom.borrow().powf(2.0) * 100.0 + 30.0).clamp(30.0, 130.0);
        let base_direction =
            crate::Vector([0.0, -1.0, 1.0 + 2.0 * *self.zoom.borrow()]).normalize(); // Start at 45° angle

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
        dbg!(transform);
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

        use crate::{Intersect, Plane};
        let ground = Plane::Z;

        Some(ground.intersect(before)? - ground.intersect(after)?)
    }
    pub fn draw(&self) {
        let input = self.window.borrow_mut().take_input();
        for input in input {
            match input {
                Input::Pan(screen_position, screen_velocity) => {
                    dbg!(screen_position, screen_velocity);
                    let vel = self
                        .pan_screen_in_world(screen_position, screen_velocity)
                        .unwrap_or(Vector::ZERO);
                    dbg!(vel);
                    *self.camera_vel.borrow_mut() = Vector([vel[0], vel[1]]);
                }
                Input::Zoom(zoom) => {
                    *self.zoom_vel.borrow_mut() = zoom;
                }
            }
        }
        *self.angle.borrow_mut() += PI / 4.0 / 60.0;
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
