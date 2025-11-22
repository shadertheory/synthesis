use core::slice;
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
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::darwin::{
    CommandBufferProtocol, ComputePipelineProtocol, DeviceProtocol, LibraryProtocol,
    ResidencyProtocol,
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
pub struct Source {
    id: u32,
    position: Vector<3, f32>,
    volume: f32,
    occlusion: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Listener {
    position: Vector<3, f32>,
    orientation: Quaternion<f32>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Probe {
    pub outdoor: f32,
    pub delay: f32,
    pub decay: f32,
    pub ambient: Vector<3, f32>,
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
    probe: Probe,

    scan: ComputePipelineProtocol,
    occlusion: ComputePipelineProtocol,
    visualizer: ComputePipelineProtocol,

    listener_data: Vec<BufferProtocol>,
    source_data: Vec<BufferProtocol>,
    probe_data: Vec<BufferProtocol>,
    visualization_data: Vec<BufferProtocol>,
}

impl Acoustics {
    pub unsafe fn init(
        in_flight: usize,
        device: &DeviceProtocol,
        library: &LibraryProtocol,
    ) -> Self {
        let max_sources = 64;
        let max_dots = 100;
        let max_probes = 1000;

        let make_pipeline = |func: &str| {
            let desc = MTLComputePipelineDescriptor::new();
            let name = NSString::from_str(func);
            let function = library.newFunctionWithName(&name).unwrap();
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

        let scan = make_pipeline("scan");
        let occlusion = make_pipeline("occlude");
        let visualizer = make_pipeline("visualize");

        let listener_data = vec![make_buf(mem::size_of::<Listener>()); in_flight];
        let source_data = vec![make_buf(max_sources * mem::size_of::<Source>()), in_flight];
        let probe_data = vec![make_buf(max_probes * mem::size_of::<Probe>()), in_flight];
        let visualization_data = vec![make_buf(max_dots * mem::size_of::<Visualizer>()), in_flight];

        Self {
            scan,
            occlusion,
            visualizer,
            listener_data,
            source_data,
            probe_data,
            visualization_data,
            listener: None,
            probe: Default::default(),
            sources: Default::default(),
        }
    }

    pub unsafe fn dispatch(&mut self, cmd: &CommandBufferProtocol, residency: &ResidencyProtocol) {
        let enc = cmd.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(&self.scan);
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(one, one);

        enc.setComputePipelineState(&self.occlusion);
        enc.dispatchThreads_threadsPerThreadgroup(MTLSize, threads_per_threadgroup);
    }

    pub unsafe fn readback(&mut self, ring: usize) {
        let probe_data = &mut self.probe_data[ring];
        let source_data = &mut self.source_data[ring];

        let probe_count = *probe_data.contents().cast::<u32>();
        let probe_ptr = probe_data
            .contents()
            .byte_add(mem::size_of::<u32>())
            .cast::<probe>();
        let probe = (0..probe_count).map(|x| probe_ptr[x]).sum() / probe_count;

        self.probe(probe);

        let source_count = *source_data.contents().cast::<u32>();
        let source_ptr = source_data
            .contents()
            .byte_add(mem::size_of::<u32>())
            .cast::<Source>();
        let sources = slice::from_raw_parts::<Source>(source_ptr, source_count);

        for gpu in sources {
            let Ok(cpu_idx) = self.sources.binary_search_by_key(gpu.id, |src| src.id) else {
                continue;
            };

            self.sources[cpu_idx].occlusion = gpu.occlusion;
        }
    }

    pub unsafe fn upload(&mut self, ring: usize) -> bool {
        let source_count = self.sources.len();
        if source_count == 0 {
            return false;
        }
        let listener_data = &mut self.listener_data[ring];
        let source_data = &mut self.source_data[ring];

        ptr::copy_nonoverlapping(&self.listener, &mut *listener_data.contents().cast(), 1);
        ptr::copy_nonoverlapping(
            self.sources.as_ptr(),
            &mut *source_data.contents().cast(),
            source_count,
        );

        return true;
    }

    pub unsafe fn update(
        &mut self,
        frame: usize,
        in_flight: usize,
        engine: &Engine,
        cmd: &CommandBufferProtocol,
        residency: &ResidencyProtocol,
    ) {
        let idx = frame % in_flight;
        if frame >= in_flight {
            self.readback(idx);
        }
        if self.upload(idx) {
            self.dispatch(cmd, residency);
        }
        engine.apply(probe);
    }

    pub unsafe fn probe(&mut self, probe: Probe) {
        self.probe = probe;
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
            2, tf
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
            }),
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

        for i in 0..left.len() {
            let mut mix = 0.0;
            for voice in &mut state.voices {
                mix += voice.render(sample_rate);
            }
            mix *= state.volume;
            left[i] += mix;
            right[i] += mix;
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
