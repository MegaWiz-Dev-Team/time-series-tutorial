use serde::Serialize;

/// A single signal channel from the recording
#[derive(Debug, Clone)]
pub struct Signal {
    pub label: String,
    pub samples: Vec<f64>,
    pub sample_rate: f64,
    pub unit: String,
}

impl Signal {
    /// Duration of the signal in seconds
    pub fn duration_sec(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate
    }

    /// Get value at a specific time (seconds), using nearest-neighbor
    pub fn value_at(&self, time_sec: f64) -> Option<f64> {
        let idx = (time_sec * self.sample_rate) as usize;
        self.samples.get(idx).copied()
    }

    /// Get a slice of samples between start_sec and end_sec
    pub fn slice_time(&self, start_sec: f64, end_sec: f64) -> &[f64] {
        let start_idx = (start_sec * self.sample_rate) as usize;
        let end_idx = ((end_sec * self.sample_rate) as usize).min(self.samples.len());
        &self.samples[start_idx.min(end_idx)..end_idx]
    }
}

/// Full recording from an HST device
#[derive(Debug)]
pub struct Recording {
    pub start_time: chrono::NaiveDateTime,
    pub duration_sec: f64,
    pub flow: Signal,       // CH0: Resp nasal @ 100 Hz
    pub effort: Signal,     // CH1: Resp thorax @ 10 Hz
    pub pulse: Signal,      // CH2: Pulse @ 1 Hz
    pub spo2: Signal,       // CH3: SaO2 @ 1 Hz
    pub position: Signal,   // CH5: Position @ 1 Hz
    pub accel_x: Signal,    // CH6: Acc x @ 10 Hz
    pub accel_y: Signal,    // CH7: Acc y @ 10 Hz
    pub accel_z: Signal,    // CH8: Acc z @ 10 Hz
}

impl Recording {
    pub fn channel_count(&self) -> usize {
        9 // fixed for ApneaLink Air
    }
}

/// Body position decoded from position channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum BodyPosition {
    Upright,
    Prone,
    Left,
    Supine,
    Right,
    Unknown,
}

impl BodyPosition {
    pub fn from_code(code: f64) -> Self {
        match code as u8 {
            1 => BodyPosition::Upright,
            2 => BodyPosition::Prone,
            3 => BodyPosition::Left,
            4 => BodyPosition::Supine,
            5 => BodyPosition::Right,
            _ => BodyPosition::Unknown,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            BodyPosition::Upright => "Upright",
            BodyPosition::Prone => "Prone",
            BodyPosition::Left => "Left",
            BodyPosition::Supine => "Supine",
            BodyPosition::Right => "Right",
            BodyPosition::Unknown => "Unknown",
        }
    }
}
