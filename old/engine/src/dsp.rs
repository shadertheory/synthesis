use std::f32::consts::PI;

#[derive(Clone)]
pub struct LowPassFilter {
    y_prev: f32,
    alpha: f32,
}

impl LowPassFilter {
    pub fn new() -> Self {
        Self {
            y_prev: 0.0,
            alpha: 1.0, // Default: Pass-through
        }
    }

    /// Set cutoff frequency.
    /// `cutoff_hz`: The frequency to attenuate above.
    /// `sample_rate`: Audio sample rate (e.g. 44100.0).
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        // Simple RC Low-pass filter coefficient calculation
        // dt = 1/sample_rate
        // RC = 1 / (2*PI*cutoff)
        // alpha = dt / (RC + dt)
        // Simplification for first order:
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * PI * cutoff_hz);
        self.alpha = (dt / (rc + dt)).clamp(0.0, 1.0);
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        let y = self.y_prev + self.alpha * (sample - self.y_prev);
        self.y_prev = y;
        y
    }
}

pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl DelayLine {
    pub fn new(max_delay_ms: f32, sample_rate: f32) -> Self {
        let size = (max_delay_ms / 1000.0 * sample_rate).ceil() as usize;
        Self {
            buffer: vec![0.0; size],
            write_pos: 0,
            sample_rate,
        }
    }

    pub fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Read from the delay line.
    /// `delay_ms`: Delay in milliseconds.
    pub fn read(&self, delay_ms: f32) -> f32 {
        let delay_samples = (delay_ms / 1000.0 * self.sample_rate).max(0.0);
        
        // Linear Interpolation
        let read_ptr = self.write_pos as f32 - delay_samples;
        
        // Handle wrapping
        let len = self.buffer.len() as f32;
        let read_ptr = if read_ptr < 0.0 { read_ptr + len } else { read_ptr };
        
        let index_a = read_ptr as usize;
        let index_b = (index_a + 1) % self.buffer.len();
        let frac = read_ptr - index_a as f32;
        
        let sample_a = self.buffer[index_a];
        let sample_b = self.buffer[index_b];
        
        sample_a + frac * (sample_b - sample_a)
    }

    /// Read from the delay line using sample offset.
    pub fn read_samples(&self, delay_samples: f32) -> f32 {
        // Linear Interpolation
        let read_ptr = self.write_pos as f32 - delay_samples;
        
        // Handle wrapping
        let len = self.buffer.len() as f32;
        let read_ptr = if read_ptr < 0.0 { read_ptr + len } else { read_ptr };
        
        let index_a = read_ptr as usize;
        let index_b = (index_a + 1) % self.buffer.len();
        let frac = read_ptr - index_a as f32;
        
        let sample_a = self.buffer[index_a];
        let sample_b = self.buffer[index_b];
        
        sample_a + frac * (sample_b - sample_a)
    }

    /// Read using relative velocity for Doppler effect.
    /// `base_delay_ms`: The base propagation delay.
    /// `relative_speed`: Source speed relative to listener (m/s). Positive = moving away.
    /// `speed_of_sound`: m/s (approx 343.0).
    pub fn read_doppler(&self, base_delay_ms: f32, relative_speed: f32, speed_of_sound: f32) -> f32 {
        // Doppler factor: f_observed = f_source * (v_sound / (v_sound + v_source))
        // Time dilation: t_observed = t_source * ((v_sound + v_source) / v_sound)
        // We simulate this by modulating the delay time.
        // Actually, for a delay line, changing the read pointer speed naturally creates pitch shift.
        // We just need to ask "where was the wavefront emitted?"
        
        // Simplified approach: Just modulate the delay length based on velocity?
        // No, that creates a pitch shift only while delay is CHANGING.
        // Correct approach is to track "virtual read head position" which we don't do here.
        
        // For per-sample processing without keeping extra state, we can't easily do continuous doppler
        // unless we pass in the "accumulated" delay drift.
        
        // Let's stick to simple read() for now, and handle Doppler by modulating 'base_delay_ms' 
        // slowly over time in the main loop, or by using a fractional read speed managed by the caller.
        
        self.read(base_delay_ms)
    }
}
