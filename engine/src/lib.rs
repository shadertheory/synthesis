#![feature(
    trait_alias,
    thread_local,
    inherent_associated_types,
    generic_const_exprs,
    associated_type_defaults
)]

use std::{
    cell::RefCell,
    convert::identity,
    fmt,
    iter::Sum,
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

pub mod darwin {
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
        metal_device: Retained<ProtocolObject<dyn MTLDevice>>,
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
        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
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
        pub unsafe fn next(
            &mut self,
        ) -> (Retained<ProtocolObject<dyn MTLTexture + 'static>>, Size) {
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
        metal_device: Retained<ProtocolObject<dyn MTLDevice>>,
        bounds: Size,
    }
    #[cfg(target_os = "macos")]
    impl Surface {
        pub unsafe fn new() -> Surface {
            let view_controller: UIViewController =
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
        pub fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.surface.device()
        }
        pub fn next(&mut self) -> (Retained<ProtocolObject<dyn MTLTexture + 'static>>, Size) {
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
    static mut LATEST_CMDS: Option<Retained<ProtocolObject<dyn MTL4CommandBuffer + 'static>>> =
        None;
    pub struct Metal {
        frame: usize,
        frame_in_flight: usize,
        render_size: Size,
        render_texture: Vec<Retained<ProtocolObject<dyn MTLTexture + 'static>>>,
        next_texture: Box<dyn Fn() -> (Retained<ProtocolObject<dyn MTLTexture + 'static>>, Size)>,
        next_present: Option<Box<dyn Fn(&mut Metal)>>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
        command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
        command_allocator: Vec<Retained<ProtocolObject<dyn MTL4CommandAllocator>>>,
        instance_accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
        frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        ui_raytrace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        argument_table: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
        argument_table2: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
        camera_buffer: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        visible_function_table: Retained<ProtocolObject<dyn MTLVisibleFunctionTable>>,
        upscale: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        raytrace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    impl Metal {
        unsafe fn init(
            device: Retained<ProtocolObject<dyn MTLDevice>>,
            next_texture: Box<
                dyn Fn() -> (Retained<ProtocolObject<dyn MTLTexture + 'static>>, Size),
            >,
            next_present: Box<dyn Fn(&mut Metal)>,
        ) -> Self {
            let frame_in_flight = 3;

            let next_present = Some(next_present);

            if !device.supportsFamily(MTLGPUFamily::Metal4) {
                panic!("Metal 4 not supported.");
            }

            let command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>> =
                msg_send![&device, newMTL4CommandQueue];
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
                let intersect_desc = MTLIntersectionFunctionDescriptor::init(
                    MTLIntersectionFunctionDescriptor::alloc(),
                );
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
                let linked = MTLLinkedFunctions::new();
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
            let residency_desc =
                MTLResidencySetDescriptor::init(MTLResidencySetDescriptor::alloc());

            let residency_set = device
                .newResidencySetWithDescriptor_error(&residency_desc)
                .unwrap();

            for i in 0..frame_in_flight {
                let _: () = msg_send![&*residency_set, addAllocation: &*render_texture[i]];
                let _: () = msg_send![&*residency_set, addAllocation: &*camera_buffer[i]];
            }

            let bounding_box_buffer = device
                .newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared)
                .unwrap();
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
            let instance_count_buffer = device
                .newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared)
                .unwrap();

            *instance_count_buffer.contents().cast::<u32>().as_mut() = 1;

            let visible_function_table_desc = MTLVisibleFunctionTableDescriptor::new();
            visible_function_table_desc.setFunctionCount(1);

            let visible_function_table = raytrace
                .newVisibleFunctionTableWithDescriptor(&visible_function_table_desc)
                .expect("Failed to create visible function table");

            // Get the intersection function handle
            let intersect_name = NSString::from_str("sphere_intersect");
            let intersect_function_handle = raytrace
                .functionHandleWithFunction(&library.newFunctionWithName(&intersect_name).unwrap())
                .expect("Failed to get function handle");

            // Set it in the table at index 0
            visible_function_table.setFunction_atIndex(Some(&intersect_function_handle), 0);

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

            let accel_desc = MTL4PrimitiveAccelerationStructureDescriptor::init(
                MTL4PrimitiveAccelerationStructureDescriptor::alloc(),
            );

            let geometry_array = NSArray::from_slice(&[&*geometry]);
            let _: () = msg_send![&*accel_desc, setGeometryDescriptors:&*geometry_array];
            let prim_accel =
                create_accel(&device, &command_queue, &command_allocator[0], &&accel_desc);

            let instance_buffer = device
                .newBufferWithLength_options(512, MTLResourceOptions::StorageModeShared)
                .unwrap();

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

            let mut instance_desc = MTL4IndirectInstanceAccelerationStructureDescriptor::init(
                MTL4IndirectInstanceAccelerationStructureDescriptor::alloc(),
            );

            instance_desc.setMaxInstanceCount(1);
            instance_desc.setInstanceCountBuffer(MTL4BufferRange {
                bufferAddress: instance_count_buffer.gpuAddress(),
                length: 32,
            });
            instance_desc.setInstanceDescriptorBuffer(MTL4BufferRange {
                bufferAddress: instance_buffer.gpuAddress(),
                length: 512,
            });
            instance_desc.setInstanceDescriptorType(
                MTLAccelerationStructureInstanceDescriptorType::Indirect,
            );

            let instance_accel = create_accel(
                &device,
                &command_queue,
                &command_allocator[0],
                &instance_desc,
            );
            unsafe fn create_accel(
                device: &ProtocolObject<dyn MTLDevice>,
                queue: &ProtocolObject<dyn MTL4CommandQueue>,
                accel_allocator: &ProtocolObject<dyn MTL4CommandAllocator>,
                accel_desc: &MTL4AccelerationStructureDescriptor,
            ) -> Retained<ProtocolObject<dyn MTLAccelerationStructure>> {
                let accel_size = device.accelerationStructureSizesWithDescriptor(accel_desc);

                let accel_struct = device
                    .newAccelerationStructureWithSize(accel_size.accelerationStructureSize)
                    .unwrap();

                let accel_scratch = device
                    .newBufferWithLength_options(
                        accel_size.buildScratchBufferSize,
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .unwrap();

                accel_allocator.reset();
                let command_buffer = device.newCommandBuffer().unwrap();

                command_buffer.beginCommandBufferWithAllocator(accel_allocator);

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
                command_buffer.endCommandBuffer();
                let cmd_ptr = NonNull::from(&*command_buffer);
                let mut cmd_array = [cmd_ptr];
                let cmd_array_ptr = NonNull::new_unchecked(cmd_array.as_mut_ptr());

                queue.commit_count(cmd_array_ptr, 1);

                accel_struct
            }

            let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
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

            let _: () = msg_send![&*residency_set, addAllocation: &*prim_accel];
            let _: () = msg_send![&*residency_set, addAllocation: &*instance_accel];
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
                instance_accel,
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
                visible_function_table,
            }
        }

        unsafe fn render(&mut self, camera: crate::CameraData) {
            if self.frame >= self.frame_in_flight {
                let frame_to_wait = self.frame - self.frame_in_flight;
                let done: bool = msg_send![&self.frame_event, waitUntilSignaledValue:frame_to_wait, timeoutMS:10usize];
            }

            let frame_index = self.frame % self.frame_in_flight;
            self.frame += 1;

            let (id, size) = (self.next_texture)();

            let command_buffer = self.device.newCommandBuffer().unwrap();
            self.command_allocator[frame_index].reset();
            command_buffer.beginCommandBufferWithAllocator(&self.command_allocator[frame_index]);
            self.argument_table[frame_index]
                .setTexture_atIndex(self.render_texture[frame_index].gpuResourceID(), 0);
            self.argument_table[frame_index]
                .setAddress_atIndex(self.camera_buffer[frame_index].gpuAddress(), 0);
            self.argument_table[frame_index]
                .setResource_atBufferIndex(self.instance_accel.gpuResourceID(), 1);
            self.argument_table[frame_index]
                .setResource_atBufferIndex(self.visible_function_table.gpuResourceID(), 2);

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
                MTLStages::Dispatch,
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

            let render_descriptor =
                MTL4RenderPassDescriptor::init(MTL4RenderPassDescriptor::alloc());
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
                zoom: 0.0.into(),
                zoom_vel: 0.0.into(),
            }
        }
        pub fn camera_data(&self) -> crate::CameraData {
            use crate::CameraData;
            let screen_size = self.metal.borrow().render_size;
            let native_scale = self.window.borrow().native_scale();
            let camera_pos = self.camera_pos.borrow();
            let projection = Projection {
                fov: (PI / 4.0 * *self.zoom.borrow() + PI / 6.0)
                    .clamp(PI / 6.0, PI / 6.0 + PI / 4.0),
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
            let pos = center + rotated_direction * distance;
            dbg!(center, pos, rotated_direction, distance);
            let mut transform = Quaternion::look_rotation(-rotated_direction, up).to_matrix();
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix<const ROWS: usize, const COLS: usize, T>([T; { ROWS * COLS }])
where
    [(); { ROWS * COLS }]: Sized;

macro_rules! matrix_elementwise_op {
    ($trait:tt, $func:tt, $op:tt, $op_assign:tt) => {
        impl<const ROWS: usize, const COLS: usize, T: $trait + Num> $trait<Self> for Matrix<ROWS, COLS, T>
        where
            <T as $trait>::Output: Num,
            [(); { ROWS * COLS }]: Sized,
        {
            type Output = Matrix<ROWS, COLS, <T as $trait>::Output>;

            fn $func(self, rhs: Self) -> <Self as $trait<Self>>::Output {
                let mut ret = <Self as $trait<Self>>::Output::identity();
                for i in 0..ROWS * COLS {
                    ret.0[i] = self.0[i] $op rhs.0[i];
                }
                ret
            }
        }

        paste::paste! {
            // Matrix op= Matrix
            impl<const ROWS: usize, const COLS: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<Self> for Matrix<ROWS, COLS, T>
            where
                <T as $trait>::Output: Num,
                [(); { ROWS * COLS }]: Sized,
            {
                fn [<$func _assign>](&mut self, rhs: Self) {
                    for i in 0..ROWS * COLS {
                        self.0[i] $op_assign rhs.0[i];
                    }
                }
            }
        }
    };
}

macro_rules! matrix_scalar_op {
    ($trait:tt, $func:tt, $op:tt, $op_assign:tt) => {

        impl<const ROWS: usize, const COLS: usize, T: $trait + Num> $trait<T> for Matrix<ROWS, COLS, T>
        where
            <T as $trait>::Output: Num,
            [(); { ROWS * COLS }]: Sized,
        {
            type Output = Matrix<ROWS, COLS, <T as $trait>::Output>;

            fn $func(self, scalar: T) -> <Self as $trait<T>>::Output {
                let mut ret = <Self as $trait<T>>::Output::identity();
                for i in 0..ROWS * COLS {
                    ret.0[i] = self.0[i] $op scalar;
                }
                ret
            }
        }

        paste::paste! {
            // Matrix op= scalar
            impl<const ROWS: usize, const COLS: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<T> for Matrix<ROWS, COLS, T>
            where
                <T as $trait>::Output: Num,
                [(); { ROWS * COLS }]: Sized,
            {
                fn [<$func _assign>](&mut self, scalar: T) {
                    for i in 0..ROWS * COLS {
                        self.0[i] $op_assign scalar;
                    }
                }
            }
        }
    };
}

// Apply macros
matrix_elementwise_op!(Add, add, +, +=);
matrix_elementwise_op!(Sub, sub, -, -=);
matrix_scalar_op!(Mul, mul, *, *=);
matrix_scalar_op!(Div, div, /, /=);

// Negation (unary operator)
impl<const ROWS: usize, const COLS: usize, T> Neg for Matrix<ROWS, COLS, T>
where
    T: Neg + Num,
    <T as Neg>::Output: Num,
    [(); { ROWS * COLS }]: Sized,
{
    type Output = Matrix<ROWS, COLS, <T as Neg>::Output>;

    fn neg(self) -> <Self as Neg>::Output {
        let mut result = <Self as Neg>::Output::identity();
        for i in 0..ROWS * COLS {
            result.0[i] = -self.0[i];
        }
        result
    }
}
impl<T: Float> Mul<Vector<3, T>> for Quaternion<T> {
    type Output = Vector<3, T>;
    fn mul(self, rhs: Vector<3, T>) -> <Self as Mul<Vector<3, T>>>::Output {
        let Quaternion {
            scalar: w,
            vector: u,
        } = self;
        let v = rhs;
        let two = T::ONE + T::ONE;
        u * (two * u.dot(v)) + v * (w * w - u.magnitude_squared()) + u.cross(v) * (two * w)
    }
}
// Matrix multiplication (special case - can't use macro)
impl<const ROWS: usize, const COLS: usize, T> Mul<Vector<COLS, T>> for Matrix<ROWS, COLS, T>
where
    T: Num,
    [(); { ROWS * COLS }]: Sized,
    [(); { ROWS }]: Sized,
    [(); { COLS }]: Sized,
{
    type Output = Vector<COLS, T>;

    fn mul(self, rhs: Vector<COLS, T>) -> <Self as Mul<Vector<COLS, T>>>::Output {
        let mut result = Vector::ZERO;

        for row in 0..ROWS {
            let mut sum = T::ZERO;
            for col in 0..COLS {
                sum = sum + self[(row, col)] * rhs[col];
            }
            result[row] = sum;
        }

        result
    }
}
impl<const M: usize, const N: usize, const P: usize, T> Mul<Matrix<N, P, T>> for Matrix<M, N, T>
where
    T: Float,
    [(); { M * N }]: Sized,
    [(); { N * P }]: Sized,
    [(); { M * P }]: Sized,
{
    type Output = Matrix<M, P, T>;

    fn mul(self, rhs: Matrix<N, P, T>) -> <Self as Mul<Matrix<N, P, T>>>::Output {
        let mut result = [T::ZERO; M * P];

        for i in 0..M {
            // For each row i
            for j in 0..P {
                // For each col j
                let mut sum = T::ZERO;
                for k in 0..N {
                    // For each common dim k
                    let self_val = self.0[k * M + i]; // self[i,k]
                    let rhs_val = rhs.0[j * N + k]; // rhs[k,j]
                    sum = sum + self_val * rhs_val;
                }
                result[j * M + i] = sum;
            }
        }
        Matrix(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T: Num> FromIterator<T> for Matrix<ROWS, COLS, T>
where
    [(); { ROWS * COLS }]: Sized,
{
    fn from_iter<IntoIter: IntoIterator<Item = T>>(iter: IntoIter) -> Self {
        let mut matrix = Self::identity();
        for (index, num) in (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row, col)))
            .zip(iter)
        {
            matrix[index] = num;
        }
        matrix
    }
}

impl<const ROWS: usize, const COLS: usize, T: Num> Matrix<ROWS, COLS, T>
where
    [(); { ROWS * COLS }]: Sized,
{
    const ZERO: Self = Self([T::ZERO; { ROWS * COLS }]);
    fn identity() -> Self {
        let mut arr = [T::ZERO; { ROWS * COLS }];
        for index in 0..ROWS * COLS {
            if index % (ROWS + 1) == 0 {
                arr[index] = T::ONE;
            };
        }
        Self(arr)
    }

    fn from_translation(translation: Vector<{ COLS - 1 }, T>) -> Self {
        let mut matrix = Matrix::identity();
        for row in 0..ROWS - 1 {
            let dim = row;
            matrix[(row, COLS - 1)] = translation[dim];
        }
        matrix
    }

    fn transpose(self) -> Self {
        (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row, col)))
            .map(|(row, col)| self[(col, row)])
            .collect()
    }
    fn minor(self, skip_row: usize, skip_col: usize) -> Matrix<{ ROWS - 1 }, { COLS - 1 }, T>
    where
        [(); { (ROWS - 1) * (COLS - 1) }]: Sized,
    {
        (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row, col)))
            .filter(|&(row, col)| row != skip_row && col != skip_col)
            .map(|index| self[index])
            .collect()
    }

    fn from_array(array: [T; { ROWS * COLS }]) -> Self {
        Self(array)
    }
    fn to_array(self) -> [T; { ROWS * COLS }]
    where
        [(); { ROWS * COLS }]: Sized,
    {
        self.0
    }

    fn determinant(&self) -> T
    where
        T: Signed,
    {
        fn two_by_two<T: Num>(this: Matrix<2, 2, T>) -> T {
            this[(0, 0)] * this[(1, 1)] - this[(0, 1)] * this[(1, 0)]
        }
        if ROWS == 2 && COLS == 2 {
            return (two_by_two)(self.to_array().into_iter().collect());
        }
        if ROWS != COLS {
            panic!("determinant can only be taken on square matrices");
        }
        let mut stack = vec![(self.to_array().to_vec(), ROWS, T::ONE)];
        let mut result = T::ZERO;
        while let Some((matrix, num_rows, coef)) = stack.pop() {
            let num_cols = num_rows;
            if num_rows == 2 {
                result += coef * two_by_two(matrix.into_iter().collect());
            } else {
                for col in 0..num_rows {
                    let element = matrix[col];
                    let sign = if col % 2 == 0 { T::ONE } else { T::ONE.neg() };

                    let mut minor = vec![];
                    for row in 1..num_rows {
                        for skip_col in 0..num_rows {
                            if skip_col != col {
                                let col = skip_col;
                                minor.push(matrix[row * num_cols + col]);
                            }
                        }
                    }

                    stack.push((minor, num_rows - 1, coef * sign * element))
                }
            }
        }
        result
    }
    fn inverse(self) -> Option<Self>
    where
        [(); (ROWS - 1) * (COLS - 1)]: Sized,
        T: Signed + fmt::Debug,
    {
        if ROWS != COLS {
            panic!("inverse can only be taken on square matrices");
        }

        let det = self.determinant();

        if det == T::ZERO {
            return None;
        }

        if ROWS == 2 {
            let mut result = Self::identity();
            result[(0, 0)] = self[(1, 1)] / det;
            result[(0, 1)] = (-T::ONE * self[(0, 1)]) / det;
            result[(1, 0)] = (-T::ONE * self[(1, 0)]) / det;
            result[(1, 1)] = self[(0, 0)] / det;
            return Some(result);
        }

        let mut cofactors = Self::identity();

        for (row, col) in (0..ROWS).flat_map(|row| (0..COLS).map(move |col| (row, col))) {
            let minor_det = self.minor(row, col).determinant();

            let sign = if (row + col) % 2 == 0 {
                T::ONE
            } else {
                -T::ONE
            };
            cofactors[(row, col)] = sign * minor_det;
        }

        let adjugate = cofactors.transpose();

        let mut result = Self::identity();
        for index in (0..ROWS).flat_map(|row| (0..COLS).map(move |col| (row, col))) {
            result[index] = adjugate[index] / det;
        }
        Some(result)
    }
}

type RowCol = (usize, usize);
impl<const ROWS: usize, const COLS: usize, T> Index<RowCol> for Matrix<ROWS, COLS, T>
where
    [T; { ROWS * COLS }]: Sized,
{
    type Output = T;

    fn index(&self, index: RowCol) -> &<Self as Index<RowCol>>::Output {
        let (row, col) = index;
        &self.0[col * ROWS + row]
    }
}

impl<const ROWS: usize, const COLS: usize, T> IndexMut<RowCol> for Matrix<ROWS, COLS, T>
where
    [T; { ROWS * COLS }]: Sized,
{
    fn index_mut(&mut self, index: RowCol) -> &mut <Self as Index<RowCol>>::Output {
        let (row, col) = index;
        &mut self.0[col * ROWS + row]
    }
}

#[derive(Clone, Copy)]
pub struct Projection {
    near: f32,
    fov: f32,
    aspect: f32,
}

impl Projection {
    fn to_matrix(self) -> Matrix<4, 4, f32> {
        let angle = (self.fov / 2.0);
        use crate::Trig;
        let f = 1.0 / (self.fov / 2.0).cot();
        let mut matrix = Matrix::ZERO;

        // Column 0
        matrix[(0, 0)] = f / self.aspect;

        // oolumn 1
        matrix[(1, 1)] = f;

        // Column 2
        matrix[(3, 2)] = self.near;

        // Column 3
        matrix[(2, 3)] = -1.0;

        matrix
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vector<const DIM: usize, T>([T; DIM]);

#[derive(Debug, Clone, Copy)]
pub struct Raycast<const DIM: usize, T> {
    origin: Vector<DIM, T>,
    direction: Vector<DIM, T>,
}

impl<const DIM: usize, T: Num> Raycast<DIM, T> {
    fn at(self, distance: T) -> Vector<DIM, T> {
        self.origin + self.direction * distance
    }
}

pub trait Intersect<const DIM: usize> {
    type Scalar;
    fn intersect(
        &self,
        ray: Raycast<DIM, <Self as Intersect<DIM>>::Scalar>,
    ) -> Option<Vector<DIM, <Self as Intersect<DIM>>::Scalar>>;
}

pub struct Plane {
    origin: Vector<3, f32>,
    normal: Vector<3, f32>,
}

impl Plane {
    const Z: Self = Self {
        origin: Vector::ZERO,
        normal: Vector::<3, f32>::Z,
    };
}

impl Intersect<3> for Plane {
    type Scalar = f32;

    fn intersect(
        &self,
        ray: Raycast<3, <Self as Intersect<3>>::Scalar>,
    ) -> Option<Vector<3, <Self as Intersect<3>>::Scalar>> {
        let denominator = (0..3).fold(0.0, |accum, i| accum + self.normal[i] * ray.direction[i]);

        if denominator == 0.0 {
            None?
        }

        let numerator = (0..3).fold(0.0, |accum, i| {
            accum + self.normal[i] * (self.origin[i] - ray.origin[i])
        });

        let distance = numerator / denominator;

        if distance < 0.0 {
            None?
        }

        Some(ray.at(distance))
    }
}

impl<const DIM: usize, T> Tensor for Vector<DIM, T> {
    type Repr = T;
}

impl<const DIM: usize, T: Num> Vector<DIM, T> {
    const ZERO: Self = Self([T::ZERO; DIM]);

    fn to_array(self) -> [T; DIM] {
        self.0
    }
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
    const ZERO: Self;
}
pub trait One {
    const ONE: Self;
}

macro_rules! primitive {
    ($type:tt, $zero:tt, $one:tt, $pow:tt) => {
        impl Zero for $type {
            const ZERO: Self = $zero;
        }

        impl One for $type {
            const ONE: Self = $one;
        }

        impl Pow for $type {
            fn pow(self, exponent: i32) -> Self {
                $type::$pow(self, exponent as _)
            }
        }
    };
}
macro_rules! float {
    ($type:tt, $pi:expr) => {
        impl Trig for $type {
            const PI: Self = $pi;
            fn cos(self) -> Self {
                $type::cos(self)
            }
            fn arc_cos(self) -> Self {
                $type::acos(self)
            }
            fn sin(self) -> Self {
                $type::sin(self)
            }
            fn arc_sin(self) -> Self {
                $type::asin(self)
            }
            fn tan(self) -> Self {
                $type::tan(self)
            }
            fn cot(self) -> Self {
                $type::cos(self) / $type::sin(self)
            }
            fn arc_tan(self) -> Self {
                $type::atan(self)
            }
            fn arc_tan2(self, other: Self) -> Self {
                $type::atan2(self, other)
            }
        }
        impl Epsilon for $type {
            const EPSILON: $type = 0.000001;
        }
        impl Sqrt for $type {
            fn sqrt(self) -> Self {
                self.sqrt()
            }
        }
    };
}
primitive!(usize, 0, 1, pow);
primitive!(u32, 0, 1, pow);
primitive!(f32, 0.0, 1.0, powi);
float!(f32, 3.1415926535);

#[derive(Clone, Copy, Debug)]
pub struct Quaternion<T> {
    pub scalar: T,
    pub vector: Vector<3, T>,
}
impl<T: Signed> Neg for Quaternion<T> {
    type Output = Self;

    fn neg(self) -> <Self as Neg>::Output {
        self.conjugate()
    }
}
impl<T: Float> Quaternion<T> {
    /// Creates a rotation that looks in the given direction with the specified up vector
    /// For Z-up coordinate system
    pub fn look_rotation(forward: Vector<3, T>, up: Vector<3, T>) -> Self {
        let forward = forward.normalize();

        // Build orthonormal basis
        // In Z-up: forward is the look direction, up is Z axis typically
        let vector = forward.normalize();
        let vector2 = up.cross(vector).normalize(); // right vector
        let vector3 = vector.cross(vector2); // corrected up vector

        // Build rotation matrix (column-major to match your Matrix implementation)
        // Column 0: right (vector2)
        let m00 = vector2[0];
        let m10 = vector2[1];
        let m20 = vector2[2];

        // Column 1: up (vector3)
        let m01 = vector3[0];
        let m11 = vector3[1];
        let m21 = vector3[2];

        // Column 2: forward (vector)
        let m02 = vector[0];
        let m12 = vector[1];
        let m22 = vector[2];

        // Convert rotation matrix to quaternion
        let trace = m00 + m11 + m22;
        let one = T::ONE;
        let half = one / (one + one);

        if trace > T::ZERO {
            let s = (trace + one).sqrt();
            let w = s * half;
            let s = half / s;
            let x = (m21 - m12) * s;
            let y = (m02 - m20) * s;
            let z = (m10 - m01) * s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        } else if (m00 >= m11) && (m00 >= m22) {
            let s = ((one + m00) - m11 - m22).sqrt();
            let inv_s = half / s;
            let x = half * s;
            let y = (m10 + m01) * inv_s;
            let z = (m20 + m02) * inv_s;
            let w = (m21 - m12) * inv_s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        } else if m11 > m22 {
            let s = ((one + m11) - m00 - m22).sqrt();
            let inv_s = half / s;
            let x = (m01 + m10) * inv_s;
            let y = half * s;
            let z = (m12 + m21) * inv_s;
            let w = (m02 - m20) * inv_s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        } else {
            let s = ((one + m22) - m00 - m11).sqrt();
            let inv_s = half / s;
            let x = (m02 + m20) * inv_s;
            let y = (m12 + m21) * inv_s;
            let z = half * s;
            let w = (m10 - m01) * inv_s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        }
    }
}
impl<T: Num> Quaternion<T> {
    const IDENTITY: Self = Self {
        scalar: T::ONE,
        vector: Vector::<3, T>::ZERO,
    };

    pub fn to_matrix(mut self) -> Matrix<4, 4, T>
    where
        T: Sqrt + Epsilon,
    {
        let Self {
            vector: Vector([x, y, z]),
            scalar: w,
        } = self.normalize();
        let one = T::ONE;
        let two = one + one;
        let zero = T::ZERO;
        Matrix::from_array([
            // Column 0 (right vector)
            one - two * y * y - two * z * z,
            two * x * y + two * z * w,
            two * x * z - two * y * w,
            zero,
            // Column 1 (up vector)
            two * x * y - two * z * w,
            one - two * x * x - two * z * z,
            two * y * z + two * x * w,
            zero,
            // Column 2 (forward vector)
            two * x * z + two * y * w,
            two * y * z - two * x * w,
            one - two * x * x - two * y * y,
            zero,
            // Column 3 (translation)
            zero,
            zero,
            zero,
            one,
        ])
    }
    pub fn from_direction(direction: Vector<3, T>, up: Vector<3, T>) -> Self
    where
        T: Float + Trig,
    {
        let dir = direction.normalize();

        // Calculate yaw (rotation around Z axis)
        let yaw = dir[1].arc_tan2(dir[0]);

        // Calculate pitch (rotation in the vertical plane)
        let horizontal_length = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        let pitch = dir[2].arc_tan2(horizontal_length);

        // Create quaternions for yaw and pitch.
        let q_yaw = Quaternion::from_axis_angle(
            Vector([T::ZERO, T::ZERO, T::ONE]), // Z-up axis
            yaw,
        );

        let q_pitch = Quaternion::from_axis_angle(
            Vector([-T::ONE, T::ZERO, T::ZERO]), // X axis (negative for proper pitch)
            pitch,
        );

        // Combine: yaw first, then pitch
        q_yaw * q_pitch
    }
    pub fn conjugate(mut self) -> Self
    where
        T: Signed,
    {
        self.vector = -self.vector;
        self
    }

    pub fn normalize(mut self) -> Self
    where
        T: Sqrt + Epsilon,
    {
        let mag = (self.scalar * self.scalar + self.vector.magnitude_squared()).sqrt();
        if mag > T::EPSILON {
            // A real T::EPSILON would be better
            self.scalar = self.scalar / mag;
            self.vector = self.vector / mag;
        }
        self
    }
}

impl<T: Float> Quaternion<T> {
    pub fn from_axis_angle(axis: Vector<3, T>, angle: T) -> Self {
        let half = T::ONE / (T::ONE + T::ONE);
        let half_angle = angle * half;
        Self {
            scalar: half_angle.cos(),
            vector: axis.normalize() * half_angle.sin(),
        }
    }
}

impl<T: Num> Mul<Self> for Quaternion<T>
where
    T: Signed,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> <Self as Mul<Self>>::Output {
        let Quaternion {
            scalar: w1,
            vector: Vector([x1, y1, z1]),
        } = self;
        let Quaternion {
            scalar: w2,
            vector: Vector([x2, y2, z2]),
        } = rhs;

        Self {
            scalar: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            vector: Vector([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]),
        }
    }
}

pub trait Pow {
    fn pow(self, exponent: i32) -> Self;
}

pub trait Epsilon: Sized {
    const EPSILON: Self;
}
pub trait Abs {
    fn abs(self) -> Self;
}
impl Abs for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }
}
pub trait Trig: Sized {
    const PI: Self;
    fn cos(self) -> Self;
    fn arc_cos(self) -> Self;
    fn sin(self) -> Self;
    fn arc_sin(self) -> Self;
    fn tan(self) -> Self;
    fn cot(self) -> Self;
    fn arc_tan(self) -> Self;
    fn arc_tan2(self, rhs: Self) -> Self;
}

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

pub trait Math = Sized
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Pow
    + Sum
    + PartialEq
    + PartialOrd;
pub trait Num = Copy + Zero + One + Math;
pub trait Signed = Num + Abs + Neg<Output = Self>;
pub trait Float = Num + Trig + Sqrt + Epsilon + Signed;

impl<const DIM: usize, T: Num> Default for Vector<DIM, T> {
    fn default() -> Self {
        let mut ret = unsafe { mem::zeroed::<[T; DIM]>() };
        for i in 0..DIM {
            ret[i] = T::ZERO;
        }
        Self(ret)
    }
}

macro_rules! vector_op {
    ($trait:tt, $func:tt, $op:tt, $op_assign:tt) => {
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
        paste::paste! {
            use std::ops::[<$trait Assign>];

            impl<const DIM: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<Self> for Vector<DIM, T> where <T as $trait>::Output: Num {
                fn [<$func _assign>](&mut self, rhs: Self) {
                    for i in 0..DIM {
                        self[i] $op_assign rhs[i];
                    }
                }
            }
            impl<const DIM: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<T> for Vector<DIM, T> where <T as $trait>::Output: Num {
                fn [<$func _assign>](&mut self, rhs: T) {
                    for i in 0..DIM {
                        self[i] $op_assign rhs;
                    }
                }
            }
        }
    };
}

impl<T: Num> Vector<2, T> {
    const X: Self = Self([T::ONE, T::ZERO]);
    const Y: Self = Self([T::ZERO, T::ONE]);
}
impl<T: Num> Vector<3, T> {
    const X: Self = Self([T::ONE, T::ZERO, T::ZERO]);
    const Y: Self = Self([T::ZERO, T::ONE, T::ZERO]);
    const Z: Self = Self([T::ZERO, T::ZERO, T::ONE]);
}
impl<T: Num> Vector<4, T> {
    const X: Self = Self([T::ONE, T::ZERO, T::ZERO, T::ZERO]);
    const Y: Self = Self([T::ZERO, T::ONE, T::ZERO, T::ZERO]);
    const Z: Self = Self([T::ZERO, T::ZERO, T::ONE, T::ZERO]);
    const W: Self = Self([T::ZERO, T::ZERO, T::ZERO, T::ONE]);

    fn perspective_divide(self) -> Vector<3, T> {
        Vector([self[0] / self[3], self[1] / self[3], self[2] / self[3]])
    }
}
impl<const DIM: usize, T: Num> Vector<DIM, T> {
    fn dot(self, rhs: Self) -> T {
        self.to_array()
            .into_iter()
            .zip(rhs.to_array())
            .fold(T::ZERO, |accum, (lhs, rhs)| accum + lhs * rhs)
    }
    fn cross(self, rhs: Self) -> Self
    where
        T: Signed,
    {
        assert!(DIM == 3, "Cross product is only defined for 3D vectors");

        Self(std::array::from_fn(|i| {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            self[j] * rhs[k] - self[k] * rhs[j]
        }))
    }
    fn magnitude_squared(self) -> T {
        self.dot(self)
    }
    fn magnitude(self) -> T
    where
        T: Sqrt,
    {
        self.magnitude_squared().sqrt()
    }
    fn normalize(self) -> Self
    where
        T: Sqrt,
    {
        self / self.magnitude()
    }
}

impl<const DIM: usize, T: Copy + Neg<Output = T>> Neg for Vector<DIM, T> {
    type Output = Self;
    fn neg(mut self) -> <Self as Neg>::Output {
        for i in 0..DIM {
            self[i] = -(self[i]);
        }
        self
    }
}

vector_op!(Add, add, +, +=);
vector_op!(Sub, sub, -, -=);
vector_op!(Mul, mul, *, *=);
vector_op!(Div, div, /, /=);
vector_op!(Rem, rem, %, %=);

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
