use std::{
    cell::UnsafeCell,
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use math::*;
use objc2::AnyThread;
use objc2_avf_audio::{
    AVAudioEngine, AVAudioFormat, AVAudioPCMBuffer, AVAudioSession, AVAudioSessionCategoryAmbient,
    AVAudioTime,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ComputeCommandEncoder, MTLComputePipelineDescriptor, MTLLibrary, MTLResourceOptions,
};

use crate::darwin::{
    BufferProtocol, CommandBufferProtocol, ComputePipelineProtocol, DeviceProtocol,
    FunctionProtocol, LibraryProtocol,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Source {
    position: Vector<3, f32>,
    volume: f32,
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
    sources: HashMap<u64, Source>,
    listener: Option<Listener>,

    scan: ComputePipelineProtocol,
    occlusion: ComputePipelineProtocol,
    visualizer: ComputePipelineProtocol,

    listener_data: BufferProtocol,
    source_data: BufferProtocol,
    probe_data: BufferProtocol,
    visualization_data: BufferProtocol,
}

impl Acoustics {
    pub unsafe fn init(
        device: &Device,
        library: &LibraryProtocol,
        residency: &ResidencyProtocol,
        structure: &AccelerationStructure,
    ) {
        let max_sources = 64;
        let max_dots = 100;
        let max_probes = 1000;

        let linked = structure.linked().unwrap();

        let make_pipeline = |func| {
            let desc = MTLComputePipelineDescriptor::new();
            desc.setComputeFunction(library.newFunctionWithName(&NSString::from_str(func)));
            desc.setLinkedFunctions(Some(&*linked));
            device
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &desc,
                    MTLPipelineOption::empty(),
                    None,
                )
                .unwrap()
        };

        let make_buf = |len| {
            device.newBufferWithLength_options(len as u64, MTLResourceOptions::StorageModeShared)
        };

        let scan = (make_pipeline)("scan");
        let occlusion = (make_pipeline)("occlude");
        let visualizer = (make_pipeline)("visualize");

        let listener_data = (make_buf)(mem::size_of::<Listener>());
        let source_data = (make_buf)(max_sources * mem::size_of::<Source>());
        let probe_data = (make_buf)(max_probes * mem::size_of::<Probe>());
        let visualization_data = (make_buf)(max_dots * mem::size_of::<Visualizer>());

        Self {
            scan,
            occlusion,
            visualizer,
            listener_data,
            source_data,
            probe_data,
            visualization_data,
            listener: None,
            sources: Default::default(),
        }
    }

    pub unsafe fn probe(device: &DeviceProtocol, cmd: &CommandBufferProtocol) -> Probe {}
}

pub trait Audio {
    fn left(&self) -> &[f32];
    fn right(&self) -> &[f32];
    fn position(&self) -> usize;
    fn volume(&self) -> f32;
    fn repeat(&self) -> bool;
    fn play(&self) -> bool;
    fn render(&self, sample_rate: f32, left: &mut [f32], right: &mut [f32]);
}

pub type Driver = Retained<AVAudioEngine>;
pub type AudioMap = Arc<UnsafeCell<HashMap<u64, Arc<dyn Audio>>>>;

pub struct Engine {
    driver: Driver,
    audio: AudioMap,
    next: u64,
    sample_rate: f64,
}

impl Engine {
    pub unsafe fn init() -> Result<Self, ()> {
        let session = AVAudioSession::sharedInstance();
        session.setCategory_error(AVAudioSessionCategoryAmbient);
        session.setActive_error(true);

        let driver = AVAudioEngine::new();
        let output_format = driver.outputNode().outputFormatForBus(0);
        let sample_rate = output_format.sampleRate();

        let format = AVAudioFormat::initStandardFormatWithSampleRate_channels(
            AVAudioFormat::alloc(),
            sample_rate,
            2,
        );
        let sample_rate = sample_rate as f32;

        let audio = Arc::new(UnsafeCell::new(HashMap::new()));

        let sources = audio.clone();
        let block = Block::new(
            move |silent: *mut bool,
                  timestamp: *const AVAudioTime,
                  frames: u32,
                  output: *mut AVAudioPCMBuffer| {
                unsafe {
                    let silent = silent.as_mut().unwrap();
                    let buffer = &*output;
                    let channel_count = buffer.format().channelCount();
                    if channel_count < 2 {
                        return -1;
                    }

                    // Get the float channel data pointer
                    let channel_data = buffer.floatChannelData();
                    if channel_data.is_null() {
                        return -1;
                    }

                    // Access left and right channels
                    let channels = std::slice::from_raw_parts(channel_data.cast::<*mut f32>(), 2);
                    let left_buffer =
                        std::slice::from_raw_parts_mut(channels[0], frame_count as usize);
                    let right_buffer =
                        std::slice::from_raw_parts_mut(channels[1], frame_count as usize);

                    left_buffer.fill(0.0);
                    right_buffer.fill(0.0);

                    for audio in audio.values() {
                        sources.render(left, right, sample_rate);
                    }
                }
            },
        );

        Self {
            driver,
            audio,
            next: 0,
            sample_rate,
        }
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
}

#[derive(Debug, Clone, Copy)]
pub enum Waveform {
    Sine,
    Square,
    Saw,
    Triangle,
    Noise,
}

impl Waveform {
    /// Sample waveform at given phase [0.0, 1.0] -> [-1.0, 1.0]
    pub fn sample(&self, phase: f32) -> f32 {
        match self {
            Waveform::Sine => (phase * std::f32::consts::TAU).sin(),
            Waveform::Square => {
                if phase % 1.0 < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Saw => 2.0 * (phase % 1.0) - 1.0,
            Waveform::Triangle => {
                let p = phase % 1.0;
                if p < 0.5 {
                    4.0 * p - 1.0
                } else {
                    3.0 - 4.0 * p
                }
            }
            Waveform::Noise => (UNIX_EPOCH
                .duration_since(SystemTime::now())
                .unwrap()
                .as_nanos() as f32)
                .sin(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    pub attack: Duration,    // seconds
    pub decay: Duration,     // seconds
    pub release: Duration,   // seconds
    pub sustain: Normalized, // 0.0 to 1.0
}

#[derive(Debug, Clone, Copy)]
pub struct Voice {
    frequency: Frequency,
    phase: f32,
    start: Instant,
    release: Option<Instant>,
    waveform: Waveform,
    envelope: Envelope,
    volume: Normalized,
    pan: Bipolar, // -1.0 (left) to 1.0 (right)
}

#[derive(Debug, Clone, Copy)]
pub struct Synthesizer {
    voices: Vec<Voice>,
    volume: Normalized,
    polyphony: usize,
}
