use core::slice;
use derive_more::{Debug, *};
use objc2::{rc::*, runtime::*, *};
use objc2_av_foundation::*;
use objc2_avf_audio::*;
use objc2_core_audio_types::*;
use objc2_core_graphics::*;
use objc2_foundation::*;
use objc2_game_controller::GCController;
use objc2_metal::*;
use objc2_metal_kit::*;
use objc2_quartz_core::*;
use objc2_ui_kit::*;
use std::{
    f32::consts::PI,
    ops::{Deref, DerefMut, Div},
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::darwin::{
    ArgumentTableProtocol, CommandBufferProtocol, ComputePipelineProtocol, DeviceProtocol,
    IntersectionFunctionTableProtocol, LibraryProtocol, ResidencyProtocol,
};
use block2::{Block, RcBlock};
use math::*;
use objc2::rc::Retained;
use objc2_avf_audio::{
    AVAudioEngine, AVAudioFormat, AVAudioPCMBuffer, AVAudioSession, AVAudioSessionCategoryAmbient,
    AVAudioTime,
};
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineDescriptor, MTLResourceOptions};
use std::mem;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Decibels {
    db: f32,
    radius: f32,
}

impl Decibels {
    fn new(db: f32) -> Self {
        Self {
            db,
            radius: Self::compute_radius(db),
        }
    }

    fn compute_radius(volume: f32) -> f32 {
        5.0
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Source {
    pub position: Vector<3, f32>,
    pub color: Vector<3, f32>,
    pub volume: Decibels,
    pub id: u32,
    pub active: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Listener {
    transform: Matrix<4, 4, f32>,
    probe_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ListenerGPU {
    pub position: Vector<3, f32>,
    pub up: Vector<3, f32>,
    pub right: Vector<3, f32>,
    pub radius: f32,
}

#[repr(C)]
#[derive(Sum, Add, Clone, Copy, Default)]
pub struct Probe {
    pub escape: u32,
    pub escape_direction: Vector<3, f32>,
    pub distance_traveled: f32,
}

pub struct ProbeResult {
    pub escape: Vector<3, f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Diffraction {
    pub hit_count: u32,
    pub direction: Vector<3, f32>,
    pub occlusion: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct OcclusionResultGPU {
    pub transmission_factor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Phonon {
    pub position: Vector<3, f32>,
    pub direction: Vector<3, f32>,
    pub energy: f32,
    pub total_distance: f32,
    pub source_id: u32,
    pub active: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct PhononHit {
    pub total_distance: f32,
    pub energy: f32,
    pub direction: Vector<3, f32>,
    pub source_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Visualizer {
    pub position: Vector<3, f32>,
    pub normal: Vector<3, f32>,
    pub size: f32,
    pub active: u32,
}

pub struct Acoustics {
    sources: Vec<Source>,
    listener: Option<Listener>,
    diffraction: Vec<Diffraction>,
    occlusion_results: Vec<OcclusionResultGPU>,
    probe: Probe,

    scan: ComputePipelineProtocol,
    occlusion: ComputePipelineProtocol,
    propagate_phonons: ComputePipelineProtocol,
    visualizer: ComputePipelineProtocol,

    listener_data: Vec<BufferProtocol>,
    source_data: Vec<BufferProtocol>,
    probe_data: Vec<BufferProtocol>,
    diffraction_data: Vec<BufferProtocol>,
    visualization_data: Vec<BufferProtocol>,

    // New Buffers
    occlusion_data: Vec<BufferProtocol>,
    phonon_buffer: BufferProtocol, // Single buffer for persistent simulation
    phonon_hits: Vec<BufferProtocol>,
    phonon_counter: Vec<BufferProtocol>,
}

impl Acoustics {
    pub unsafe fn init(
        in_flight: usize,
        device: &DeviceProtocol,
        library: &LibraryProtocol,
    ) -> Self {
        let max_sources = 64;
        let max_dots = 100;
        let max_probes = 256;

        let max_phonons = 1_000; // 10k particles
        let max_hits = 1024; // Max hits per frame to process

        let make_pipeline = |func: &str| {
            let desc = MTLComputePipelineDescriptor::new();
            let name = NSString::from_str(func);
            let function = library
                .newFunctionWithName(&name)
                .expect(&format!("Could not load {name}"));
            desc.setComputeFunction(Some(&function));
            device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    MTLPipelineOption::empty(),
                    None,
                )
                .unwrap()
        };

        let make_buf = |len| {
            device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .unwrap()
        };

        let make_private_buf = |len| {
            device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
                .unwrap()
        };

        let scan = make_pipeline("scan");
        let occlusion = make_pipeline("calc_occlusion");
        let propagate_phonons = make_pipeline("propagate_phonons");
        let visualizer = make_pipeline("visualize");

        let listener_data = vec![make_buf(mem::size_of::<ListenerGPU>()); in_flight];
        let source_data = vec![make_buf(max_sources * mem::size_of::<Source>() + 4); in_flight];
        let probe_data = vec![make_buf(max_probes * mem::size_of::<Probe>() + 4); in_flight];
        let diffraction_data =
            vec![make_buf(max_probes * max_sources * mem::size_of::<Diffraction>()); in_flight];
        let visualization_data = vec![make_buf(max_dots * mem::size_of::<Visualizer>()); in_flight];

        // New Buffers
        let occlusion_data =
            vec![make_buf(max_sources * mem::size_of::<OcclusionResultGPU>()); in_flight];

        let phonon_buffer = device
            .newBufferWithLength_options(
                max_phonons * mem::size_of::<Phonon>(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        unsafe {
            ptr::write_bytes(
                phonon_buffer.contents().as_ptr(),
                0,
                max_phonons * mem::size_of::<Phonon>(),
            );
        }

        let phonon_hits = vec![make_buf(max_hits * mem::size_of::<PhononHit>()); in_flight];
        let phonon_counter = vec![make_buf(mem::size_of::<u32>()); in_flight];

        Self {
            scan,
            occlusion,
            propagate_phonons,
            visualizer,
            listener_data,
            source_data,
            probe_data,
            diffraction_data,
            visualization_data,
            occlusion_data,
            phonon_buffer,
            phonon_hits,
            phonon_counter,
            listener: None,
            probe: Default::default(),
            sources: Default::default(),
            diffraction: vec![Diffraction::default(); max_probes * max_sources],
            occlusion_results: vec![OcclusionResultGPU::default(); max_sources],
        }
    }

    pub unsafe fn dispatch(
        &mut self,
        ring: usize,
        cmd: &CommandBufferProtocol,
        argument_table: &ArgumentTableProtocol,
    ) {
        let Some(listener) = self.listener else {
            return;
        };
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setArgumentTable(Some(argument_table));

        // Manually bind RTX resources (since ArgumentTable binding might be implicit/failing context)
        argument_table.setAddress_atIndex(self.source_data[ring].gpuAddress(), 8);
        // Listener buffer: Direct
        argument_table.setAddress_atIndex(self.listener_data[ring].gpuAddress(), 9);
        // Probe buffer: Full buffer (Header + Data)
        argument_table.setAddress_atIndex(self.probe_data[ring].gpuAddress(), 10);
        // Diffraction buffer: Direct
        argument_table.setAddress_atIndex(self.diffraction_data[ring].gpuAddress(), 11);

        // New Buffers
        argument_table.setAddress_atIndex(self.occlusion_data[ring].gpuAddress(), 12);
        // Bind Single Persistent Phonon Buffer
        argument_table.setAddress_atIndex(self.phonon_buffer.gpuAddress(), 13);
        argument_table.setAddress_atIndex(self.phonon_hits[ring].gpuAddress(), 14);
        argument_table.setAddress_atIndex(self.phonon_counter[ring].gpuAddress(), 15);

        // 2. Occlusion (Direct Path)
        enc.setComputePipelineState(&self.occlusion);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: self.sources.len(),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1, // 1 Thread per source
                height: 1,
                depth: 1,
            },
        );

        // 3. Phonons (Indirect Path)
        // 10,000 threads
        enc.setComputePipelineState(&self.propagate_phonons);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: 1000,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        enc.endEncoding();
    }

    pub unsafe fn readback(&mut self, ring: usize) {
        let probe_data = &mut self.probe_data[ring];
        let source_data = &mut self.source_data[ring];
        let diffraction_data = &mut self.diffraction_data[ring];

        // Read count from first 4 bytes
        let probe_count = *probe_data.contents().cast::<u32>().as_ptr();
        let probe_ptr = probe_data
            .contents()
            .byte_add(mem::size_of::<u32>())
            .cast::<Probe>()
            .as_ptr();

        // Manual averaging of probes
        let mut sum_probe = Probe::default();
        if probe_count > 0 {
            for i in 0..probe_count {
                let p = probe_ptr.add(i as _).read();
                sum_probe.escape += p.escape;
                sum_probe.escape_direction = sum_probe.escape_direction + p.escape_direction;
                sum_probe.distance_traveled += p.distance_traveled;
            }
            sum_probe.escape /= probe_count;
            sum_probe.escape_direction = sum_probe.escape_direction / (probe_count as f32);
            sum_probe.distance_traveled /= probe_count as f32;
        }
        self.probe = sum_probe;

        let source_count = *source_data.contents().cast::<u32>().as_ptr() as _;
        let source_ptr = source_data
            .contents()
            .byte_add(mem::size_of::<u32>())
            .cast::<Source>()
            .as_ptr();
        let sources = slice::from_raw_parts::<Source>(source_ptr, source_count);

        let diffraction_count = self.diffraction.len();
        let diffraction_ptr = diffraction_data.contents().cast::<Diffraction>().as_ptr();
        let diffraction = slice::from_raw_parts::<Diffraction>(diffraction_ptr, diffraction_count);

        // Read Occlusion (Debug)
        let occ_ptr = self.occlusion_data[ring]
            .contents()
            .cast::<OcclusionResultGPU>()
            .as_ptr();
        let occlusion = (*occ_ptr).transmission_factor;

        // Read Phonon Hit Count (Debug)
        let counter_ptr = self.phonon_counter[ring].contents().cast::<u32>().as_ptr();
        let hit_count = *counter_ptr;

        if let Some(l) = self.listener {
            let pos = Vector([
                l.transform[(0, 3)],
                l.transform[(1, 3)],
                l.transform[(2, 3)],
            ]);
            if source_count > 0 {
                println!(
                    "Debug: Listener at {:?}, Source[0] at {:?} radius {:?}",
                    pos, sources[0].position, sources[0].volume.radius
                );
            }
        }

        println!(
            "Readback: sources={}, occlusion={:.3}, phonon_hits={}",
            source_count, occlusion, hit_count
        );

        for (i, gpu) in diffraction.iter().enumerate() {
            if i < self.diffraction.len() {
                self.diffraction[i] = *gpu;
            }
        }
    }

    pub unsafe fn readback_results(&mut self, ring: usize) -> (f32, Vec<PhononHit>) {
        // Read Occlusion

        let occ_ptr = self.occlusion_data[ring]
            .contents()
            .cast::<OcclusionResultGPU>()
            .as_ptr();

        let occlusion = (*occ_ptr).transmission_factor;

        // Read Phonon Hits

        let counter_ptr = self.phonon_counter[ring].contents().cast::<u32>().as_ptr();

        let hit_count = (*counter_ptr).min(1024);

        let hits_ptr = self.phonon_hits[ring]
            .contents()
            .cast::<PhononHit>()
            .as_ptr();

        let hits = slice::from_raw_parts(hits_ptr, hit_count as usize).to_vec();

        // Clear counter for next use (though upload clears usually, but atomic needs reset)

        *self.phonon_counter[ring].contents().cast::<u32>().as_mut() = 0;

        (occlusion, hits)
    }

    pub unsafe fn upload(&mut self, ring: usize) -> bool {
        let Some(listener) = self.listener else {
            return false;
        };
        let source_count = self.sources.len().min(64);
        if source_count == 0 {
            return false;
        }
        // println!("Upload: sources={}, probes={}", source_count, listener.probe_count);

        let listener_data = &mut self.listener_data[ring];
        let source_data = &mut self.source_data[ring];
        let probe_data = &mut self.probe_data[ring];

        // Upload Listener (direct copy of struct)
        let listener_gpu = ListenerGPU {
            position: Vector([
                listener.transform[(0, 3)],
                listener.transform[(1, 3)],
                listener.transform[(2, 3)],
            ]),
            up: Vector([
                listener.transform[(0, 1)],
                listener.transform[(1, 1)],
                listener.transform[(2, 1)],
            ]),
            right: Vector([
                listener.transform[(0, 0)],
                listener.transform[(1, 0)],
                listener.transform[(2, 0)],
            ]),
            radius: 0.2,
        };
        ptr::copy_nonoverlapping(
            &listener_gpu,
            listener_data.contents().cast::<ListenerGPU>().as_ptr(),
            1,
        );

        // Upload Source Count
        *source_data.contents().cast::<u32>().as_mut() = source_count as u32;
        // Upload Sources (offset by 4 bytes)
        ptr::copy_nonoverlapping(
            self.sources.as_ptr(),
            source_data.contents().byte_add(4).cast::<Source>().as_mut(),
            source_count,
        );

        // Write Probe Count to probe buffer (for readback)
        *probe_data.contents().cast::<u32>().as_mut() = listener.probe_count as u32;

        // Clear Diffraction Buffer
        let diffraction_data = &mut self.diffraction_data[ring];
        ptr::write_bytes(
            diffraction_data.contents().as_ptr(),
            0,
            diffraction_data.length() as usize,
        );

        // Clear Phonon Counter
        *self.phonon_counter[ring].contents().cast::<u32>().as_mut() = 0;

        return true;
    }

    pub unsafe fn update(
        &mut self,
        frame: usize,
        in_flight: usize,
        engine: &Engine,
        cmd: &CommandBufferProtocol,
        argument_table: &ArgumentTableProtocol,
    ) {
        let idx = frame % in_flight;
        dbg!(frame);
        if frame >= in_flight * 2 {
            self.readback(idx);
        }
        if self.upload(idx) {
            self.dispatch(idx, cmd, argument_table);
        }
    }

    pub unsafe fn add_to_residency(&self, residency: &ResidencyProtocol) {
        for buf in &self.listener_data {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
        for buf in &self.source_data {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
        for buf in &self.probe_data {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
        for buf in &self.diffraction_data {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
        for buf in &self.visualization_data {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
        for buf in &self.occlusion_data {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }

        let _: () = msg_send![residency, addAllocation: &*self.phonon_buffer];

        for buf in &self.phonon_hits {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
        for buf in &self.phonon_counter {
            let _: () = msg_send![residency, addAllocation: &**buf];
        }
    }
}
pub trait Audio: Send + Sync {
    fn render(&self, sample_rate: f32, left: &mut [f32], right: &mut [f32]);
}

use std::collections::HashMap;

pub type AudioMap = Arc<Mutex<HashMap<u64, Arc<dyn Audio>>>>;

pub struct Engine {
    _driver: Retained<AVAudioEngine>,
    audio: AudioMap,
    next: u64,
    pub sample_rate: f32,
}

impl Engine {
    pub unsafe fn init() -> Result<Self, ()> {
        let session = AVAudioSession::sharedInstance();
        let _ = session.setCategory_error(AVAudioSessionCategoryAmbient.unwrap());

        let _ = session.setActive_error(true);

        let driver = AVAudioEngine::new();
        let output_node = driver.outputNode();
        let output_format = output_node.outputFormatForBus(0);
        let sample_rate = output_format.sampleRate() as f32;

        let audio_map: AudioMap = Arc::new(Mutex::new(HashMap::new()));
        let block_audio_map = audio_map.clone();

        let renderer = RcBlock::new(
            move |_is_silence: NonNull<Bool>,
                  _timestamp: NonNull<AudioTimeStamp>,
                  frame_count: u32,
                  mut output_data: NonNull<AudioBufferList>| {
                let buffer = unsafe { output_data.as_mut() };

                let buffers = unsafe {
                    std::slice::from_raw_parts_mut(
                        buffer.mBuffers.as_mut_ptr(),
                        buffer.mNumberBuffers as usize,
                    )
                };

                if buffers.len() < 2 {
                    // We're expecting stereo, but didn't get it.
                    // Fill what we got with silence and return.
                    for buf in buffers {
                        if !buf.mData.is_null() {
                            unsafe {
                                std::slice::from_raw_parts_mut(
                                    buf.mData as *mut u8,
                                    buf.mDataByteSize as usize,
                                )
                                .fill(0)
                            };
                        }
                    }
                    return 0;
                }

                let left_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        buffers[0].mData as *mut f32,
                        frame_count as usize,
                    )
                };
                let right_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        buffers[1].mData as *mut f32,
                        frame_count as usize,
                    )
                };

                left_slice.fill(0.0);
                right_slice.fill(0.0);

                if let Ok(map) = block_audio_map.lock() {
                    for source in map.values() {
                        source.render(sample_rate, left_slice, right_slice);
                    }
                }

                0
            },
        );

        let source_node = objc2_avf_audio::AVAudioSourceNode::initWithRenderBlock(
            objc2_avf_audio::AVAudioSourceNode::alloc(),
            RcBlock::as_ptr(&renderer),
        );

        driver.attachNode(&source_node);
        let format = AVAudioFormat::initStandardFormatWithSampleRate_channels(
            AVAudioFormat::alloc(),
            output_format.sampleRate(),
            2,
        )
        .unwrap();
        driver.connect_to_format(&source_node, &output_node, Some(&format));

        if let Err(e) = driver.startAndReturnError() {
            println!("Failed to start engine: {:?}", e);
            return Err(());
        }

        Ok(Self {
            _driver: driver,
            audio: audio_map,
            next: 0,
            sample_rate,
        })
    }

    fn apply(&self, probe: Probe) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Note {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accidental {
    Natural,
    Sharp,
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Semitone(i8);

impl Semitone {
    pub const fn new(value: i8) -> Self {
        Self(value)
    }

    pub fn value(&self) -> i8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Octave(u8);

impl Octave {
    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Normalized(f32);

impl Normalized {
    /// Create normalized value, clamping to [0.0, 1.0]
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}
impl Deref for Normalized {
    type Target = f32;
    fn deref(&self) -> &<Self as Deref>::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Bipolar(f32);

impl Bipolar {
    /// Create bipolar value, clamping to [-1.0, 1.0]
    pub fn new(value: f32) -> Self {
        Self(value.clamp(-1.0, 1.0))
    }

    pub fn value(&self) -> f32 {
        self.0
    }

    /// Convert from normalized [0.0, 1.0] to bipolar [-1.0, 1.0]
    pub fn from_normalized(normalized: Normalized) -> Self {
        Self(normalized.value() * 2.0 - 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Frequency(f32);

impl Frequency {
    const A4_FREQUENCY: f32 = 440.0;

    /// Create frequency directly from Hz
    pub const fn from_hz(hz: f32) -> Self {
        Self(hz)
    }

    /// Get frequency in Hz
    pub fn hz(&self) -> f32 {
        self.0
    }

    /// Create frequency from note at octave 4 with natural accidental
    pub fn note(note: Note) -> Self {
        Self::note_accidental(note, Accidental::Natural)
    }

    /// Create frequency from note and accidental at octave 4
    pub fn note_accidental(note: Note, accidental: Accidental) -> Self {
        Self::note_accidental_octave(note, accidental, Octave::new(4))
    }

    /// Create frequency from note, accidental, and octave
    pub fn note_accidental_octave(note: Note, accidental: Accidental, octave: Octave) -> Self {
        let semitones = Self::semitones_from_a4(note, accidental, octave);
        Self::from_semitones(semitones)
    }

    /// Create frequency from semitone offset (relative to A4 = 440 Hz)
    pub fn from_semitones(semitones: Semitone) -> Self {
        Self(Self::A4_FREQUENCY * 2.0f32.powf(semitones.value() as f32 / 12.0))
    }

    /// Calculate semitones from A4 given note, accidental, and octave
    fn semitones_from_a4(note: Note, accidental: Accidental, octave: Octave) -> Semitone {
        // Semitones from C to each note (C is 0)
        let note_offset = match note {
            Note::C => 0,
            Note::D => 2,
            Note::E => 4,
            Note::F => 5,
            Note::G => 7,
            Note::A => 9,
            Note::B => 11,
        };

        // Apply accidental
        let accidental_offset = match accidental {
            Accidental::Natural => 0,
            Accidental::Sharp => 1,
            Accidental::Flat => -1,
        };

        // Calculate semitones from C4
        let semitones_from_c4 = note_offset + accidental_offset;

        // Calculate octave offset (C4 is 3 semitones below A4)
        let octave_offset = (octave.value() as i8 - 4) * 12;

        // A4 is 9 semitones above C4
        let semitones_from_a4 = semitones_from_c4 - 9 + octave_offset;

        Semitone::new(semitones_from_a4)
    }

    /// Transpose frequency by semitones
    pub fn transpose(&self, semitones: Semitone) -> Self {
        Self(self.0 * 2.0f32.powf(semitones.value() as f32 / 12.0))
    }

    /// Convert frequency to MIDI note number (fractional)
    pub fn to_midi(&self) -> f32 {
        69.0 + 12.0 * (self.0 / Self::A4_FREQUENCY).log2()
    }

    /// Create frequency from MIDI note number (fractional)
    pub fn from_midi(midi: f32) -> Self {
        Self(Self::A4_FREQUENCY * 2.0f32.powf((midi - 69.0) / 12.0))
    }

    /// Get the period in seconds
    pub fn period(&self) -> f32 {
        1.0 / self.0
    }

    /// Get wavelength in meters (assuming speed of sound = 343 m/s)
    pub fn wavelength(&self) -> f32 {
        343.0 / self.0
    }
}

// Convenient constructors
impl Note {
    pub fn frequency(self) -> Frequency {
        Frequency::note(self)
    }

    pub fn with_accidental(self, accidental: Accidental) -> Frequency {
        Frequency::note_accidental(self, accidental)
    }

    pub fn with_octave(self, octave: Octave) -> Frequency {
        Frequency::note_accidental_octave(self, Accidental::Natural, octave)
    }

    pub fn with_accidental_octave(self, accidental: Accidental, octave: Octave) -> Frequency {
        Frequency::note_accidental_octave(self, accidental, octave)
    }
    pub fn octave(&self, octave: u8) -> Frequency {
        Frequency::note_accidental_octave(*self, Accidental::Natural, Octave::new(octave))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Waveform {
    Sine,
    Square,
    Saw,
    Triangle,
}

impl Waveform {
    fn sample(&self, phase: f32) -> f32 {
        match self {
            Waveform::Sine => (phase * 2.0 * PI).sin(),
            Waveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Saw => 2.0 * phase - 1.0,
            Waveform::Triangle => {
                if phase < 0.5 {
                    4.0 * phase - 1.0
                } else {
                    3.0 - 4.0 * phase
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    pub attack: f32,  // seconds
    pub decay: f32,   // seconds
    pub sustain: f32, // 0.0 to 1.0
    pub release: f32, // seconds
}

#[derive(Debug, Clone, Copy)]
struct Voice {
    freq: f32,
    phase: f32,
    waveform: Waveform,
    envelope: Envelope,
    samples_processed: u64,
    released_at_sample: Option<u64>,
    active: bool,
    velocity: f32,
}

impl Voice {
    fn new(freq: f32, waveform: Waveform, envelope: Envelope) -> Self {
        Self {
            freq,
            phase: 0.0,
            waveform,
            envelope,
            samples_processed: 0,
            released_at_sample: None,
            active: true,
            velocity: 1.0,
        }
    }

    fn render(&mut self, sample_rate: f32) -> f32 {
        if !self.active {
            return 0.0;
        }

        let t = self.samples_processed as f32 / sample_rate;
        let amp = if t < self.envelope.attack {
            t / self.envelope.attack
        } else if t < self.envelope.attack + self.envelope.decay {
            1.0 - ((t - self.envelope.attack) / self.envelope.decay) * (1.0 - self.envelope.sustain)
        } else {
            self.envelope.sustain
        };

        let final_amp = if let Some(release_sample) = self.released_at_sample {
            let t_rel = (self.samples_processed - release_sample) as f32 / sample_rate;
            if t_rel >= self.envelope.release {
                self.active = false;
                0.0
            } else {
                amp * (1.0 - (t_rel / self.envelope.release))
            }
        } else {
            amp
        };

        let sample = self.waveform.sample(self.phase);
        self.phase = (self.phase + self.freq / sample_rate) % 1.0;
        self.samples_processed += 1;
        sample * final_amp * self.velocity
    }
}

#[derive(Debug)]
struct SynthState {
    voices: Vec<Voice>,
    volume: f32,
    pan: f32,
}

pub struct Synthesizer {
    state: Mutex<SynthState>,
}

impl Synthesizer {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(SynthState {
                voices: Vec::with_capacity(32),
                volume: 0.5,
                pan: 0.0,
            }),
        }
    }

    pub fn set_direction(&self, direction: Vector<3, f32>) {
        if let Ok(mut state) = self.state.lock() {
            // Assuming X is left/right (-1 to 1)
            // Normalize just in case
            let dir = direction.normalize();
            if !dir[0].is_nan() {
                state.pan = dir[0].clamp(-1.0, 1.0);
            }
        }
    }

    pub fn note_on(&self, freq: Frequency, wave: Waveform, env: Envelope) {
        let hz = freq.hz();
        if let Ok(mut state) = self.state.lock() {
            state.voices.push(Voice::new(hz, wave, env));
        }
    }

    pub fn note_off(&self, freq: Frequency) {
        let hz = freq.hz();
        if let Ok(mut state) = self.state.lock() {
            for voice in state
                .voices
                .iter_mut()
                .filter(|v| v.active && v.released_at_sample.is_none())
            {
                if (voice.freq - hz).abs() < 0.1 {
                    voice.released_at_sample = Some(voice.samples_processed);
                }
            }
        }
    }
}

impl Audio for Synthesizer {
    fn render(&self, sample_rate: f32, left: &mut [f32], right: &mut [f32]) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.voices.retain(|v| v.active);

        // Equal power panning with "dramatic" curve
        // pan is -1.0 (left) to 1.0 (right)
        let mut p = (state.pan + 1.0) * 0.5;

        // Apply sigmoid-like curve to exaggerate separation
        // Pushes values away from center (0.5) towards edges (0.0 and 1.0)
        p = if p < 0.5 {
            2.0 * p * p
        } else {
            1.0 - 2.0 * (1.0 - p) * (1.0 - p)
        };

        let left_gain = (1.0 - p).sqrt();
        let right_gain = p.sqrt();

        for i in 0..left.len() {
            let mut mix = 0.0;
            for voice in &mut state.voices {
                mix += voice.render(sample_rate);
            }
            mix *= state.volume;
            left[i] += mix * left_gain;
            right[i] += mix * right_gain;
        }
    }
}

use crate::dsp::{DelayLine, LowPassFilter};

pub struct SpatialAudioProcessor {
    synth: Arc<Synthesizer>,
    state: Mutex<SpatialState>,
    delay_line: Mutex<DelayLine>,
    lpf: Mutex<LowPassFilter>,
    reverb_accum: Mutex<Vec<f32>>, // Accumulator for reverb
}

struct SpatialState {
    listener_transform: Matrix<4, 4, f32>,
    source_position: Vector<3, f32>,
    source_volume: f32,
    prev_source_pos: Vector<3, f32>,
    prev_listener_pos: Vector<3, f32>,
    occlusion: f32, // 0.0 (Blocked) to 1.0 (Clear)
    hits: Vec<PhononHit>,
}

impl SpatialAudioProcessor {
    pub fn new(synth: Arc<Synthesizer>) -> Self {
        Self {
            synth,
            state: Mutex::new(SpatialState {
                listener_transform: Matrix::identity(),
                source_position: Vector::ZERO,
                source_volume: 1.0,
                prev_source_pos: Vector::ZERO,
                prev_listener_pos: Vector::ZERO,
                occlusion: 1.0,
                hits: Vec::new(),
            }),
            delay_line: Mutex::new(DelayLine::new(2000.0, 44100.0)), // Increased buffer for 2s reverb
            lpf: Mutex::new(LowPassFilter::new()),
            reverb_accum: Mutex::new(vec![0.0; 1024]), // Temp buffer
        }
    }

    pub fn update(
        &self,
        listener_transform: Matrix<4, 4, f32>,
        source_position: Vector<3, f32>,
        source_volume: f32,
        occlusion: f32,
        hits: Vec<PhononHit>,
    ) {
        if let Ok(mut state) = self.state.lock() {
            state.prev_listener_pos = Vector([
                state.listener_transform[(0, 3)],
                state.listener_transform[(1, 3)],
                state.listener_transform[(2, 3)],
            ]);
            state.prev_source_pos = state.source_position;

            state.listener_transform = listener_transform;
            state.source_position = source_position;
            state.source_volume = source_volume;
            state.occlusion = occlusion;
            state.hits = hits;
        }
    }
}

impl Audio for SpatialAudioProcessor {
    fn render(&self, sample_rate: f32, left: &mut [f32], right: &mut [f32]) {
        // ... (Source generation same as before)
        let mut temp_l = vec![0.0; left.len()];
        let mut temp_r = vec![0.0; right.len()];
        self.synth.render(sample_rate, &mut temp_l, &mut temp_r);

        let mut mono_input = vec![0.0; left.len()];
        for i in 0..left.len() {
            mono_input[i] = (temp_l[i] + temp_r[i]) * 0.5;
        }

        let (listener_transform, source_pos, source_vol, occlusion, hits) =
            if let Ok(state) = self.state.lock() {
                (
                    state.listener_transform,
                    state.source_position,
                    state.source_volume,
                    state.occlusion,
                    state.hits.clone(),
                )
            } else {
                return;
            };

        let mut delay = self.delay_line.lock().unwrap();
        let mut lpf = self.lpf.lock().unwrap();

        // --- DIRECT PATH CALCULATIONS ---
        let listener_pos = Vector([
            listener_transform[(0, 3)],
            listener_transform[(1, 3)],
            listener_transform[(2, 3)],
        ]);

        let distance = (source_pos - listener_pos).magnitude();

        // Occlusion LPF: Map 1.0 -> 20kHz, 0.0 -> 200Hz
        // Combine with distance absorption
        let dist_cutoff = 20000.0 * (-distance * 0.05).exp();
        let occ_cutoff = 200.0 + (19800.0 * occlusion * occlusion); // Quadratic falloff
        let final_cutoff = dist_cutoff.min(occ_cutoff).clamp(200.0, 20000.0);

        lpf.set_cutoff(final_cutoff, sample_rate);

        // Calculate ITD (Same as before)
        let right_vec = Vector([
            listener_transform[(0, 0)],
            listener_transform[(1, 0)],
            listener_transform[(2, 0)],
        ])
        .normalize();
        let head_radius = 0.1;
        let left_ear = listener_pos - right_vec * head_radius;
        let right_ear = listener_pos + right_vec * head_radius;
        let dist_l = (source_pos - left_ear).magnitude();
        let dist_r = (source_pos - right_ear).magnitude();
        let delay_ms_l = (dist_l / 343.0) * 1000.0;
        let delay_ms_r = (dist_r / 343.0) * 1000.0;
        let gain_l = (source_vol * occlusion) / dist_l.max(1.0);
        let gain_r = (source_vol * occlusion) / dist_r.max(1.0);

        // Pre-calculate Reverb Taps
        struct ReverbTap {
            offset_samples: f32,
            gain_l: f32,
            gain_r: f32,
        }

        let mut taps = Vec::with_capacity(hits.len());
        for hit in hits {
            let hit_delay_ms = (hit.total_distance / 343.0) * 1000.0;
            let offset_samples = (hit_delay_ms / 1000.0 * sample_rate).max(0.0);

            let pan = hit.direction.dot(right_vec).clamp(-1.0, 1.0);
            let p = (pan + 1.0) * 0.5;

            // Base reverb gain scaling
            let g_l = (1.0 - p).sqrt() * hit.energy * 0.05;
            let g_r = p.sqrt() * hit.energy * 0.05;

            taps.push(ReverbTap {
                offset_samples,
                gain_l: g_l,
                gain_r: g_r,
            });
        }

        // 4. Process Samples
        for i in 0..left.len() {
            let input = mono_input[i];

            delay.write(input);

            // Direct Path
            let direct_l = delay.read(delay_ms_l);
            let direct_r = delay.read(delay_ms_r);
            let filtered_direct = lpf.process((direct_l + direct_r) * 0.5);

            left[i] += filtered_direct * gain_l;
            right[i] += filtered_direct * gain_r;

            // Reverb Path (Full Granular)
            let mut rev_l = 0.0;
            let mut rev_r = 0.0;

            for tap in &taps {
                // Optimization: Avoid full interpolation for reverb?
                // No, aliasing sounds bad. We stick to read() logic but inline/simplify if needed.
                // delay.read uses current write_pos.
                // Since we just wrote input, write_pos is updated.
                // We need to read from (write_pos - offset).

                // We can access delay buffer directly if we make a method,
                // but calling .read() is clean. Overhead is method call + logic.
                // Ideally we'd inline this access.
                let s = delay.read_samples(tap.offset_samples);
                rev_l += s * tap.gain_l;
                rev_r += s * tap.gain_r;
            }

            left[i] += rev_l;
            right[i] += rev_r;
        }
    }
}

use crate::darwin::BufferProtocol;
impl Engine {
    pub fn add_source(&mut self, source: Arc<dyn Audio>) -> u64 {
        let id = self.next;
        self.next += 1;
        if let Ok(mut map) = self.audio.lock() {
            map.insert(id, source);
        }
        id
    }
    pub fn remove_source(&mut self, id: u64) {
        if let Ok(mut map) = self.audio.lock() {
            map.remove(&id);
        }
    }
}

impl Acoustics {
    pub fn add_source(&mut self, id: u32, position: Vector<3, f32>) {
        // Check if exists
        if let Some(src) = self.sources.iter_mut().find(|s| s.id == id) {
            src.position = position;
        } else {
            self.sources.push(Source {
                id,
                position,
                volume: Decibels::new(100.0), // Default volume
                color: Vector::ONE,
                active: 1,
            });
        }
    }

    pub fn update_listener(&mut self, transform: Matrix<4, 4, f32>) {
        self.listener = Some(Listener {
            transform,
            probe_count: 256,
        });
    }

    pub fn get_diffraction(&self, source_id: u32) -> Option<Diffraction> {
        let source_idx = self.sources.iter().position(|s| s.id == source_id)?;
        let source_count = self.sources.len();

        let probe_count = self.listener.as_ref().map(|l| l.probe_count).unwrap_or(0);
        if probe_count == 0 {
            return None;
        }

        let mut sum_diffraction = Diffraction::default();

        for p in 0..probe_count {
            let idx = p * source_count + source_idx;
            if idx < self.diffraction.len() {
                let d = self.diffraction[idx];
                sum_diffraction.hit_count += d.hit_count;
                sum_diffraction.direction = sum_diffraction.direction + d.direction;
                sum_diffraction.occlusion += d.occlusion;
            }
        }

        let scale = 1.0 / (probe_count as f32);
        sum_diffraction.direction = sum_diffraction.direction * scale;
        sum_diffraction.occlusion = sum_diffraction.occlusion * scale;
        // hit_count average
        sum_diffraction.hit_count = (sum_diffraction.hit_count as f32 * scale) as u32;

        Some(sum_diffraction)
    }

    pub fn get_source_volume(&self, source_id: u32) -> f32 {
        if let Some(src) = self.sources.iter().find(|s| s.id == source_id) {
            src.volume.db
        } else {
            1.0 // Default volume if source not found
        }
    }
}
