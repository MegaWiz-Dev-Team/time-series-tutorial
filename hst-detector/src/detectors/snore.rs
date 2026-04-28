use crate::edf::signals::{Signal, BodyPosition};
use crate::events::{SleepEvent, EventType};

/// Detect snoring events from the nasal flow signal
/// Snoring appears as high-frequency oscillations superimposed on breathing
///
/// Note: With 100 Hz sampling (Nyquist = 50 Hz), we can only detect
/// the lower-frequency components of snoring vibrations (typically 30-300 Hz)
pub fn detect(flow: &Signal, position: &Signal) -> Vec<SleepEvent> {
    let mut events = Vec::new();
    let sr = flow.sample_rate;

    // Analysis in 2-second windows with 50% overlap
    let window_samples = (2.0 * sr) as usize;
    let step = window_samples / 2;

    let mut in_snore = false;
    let mut snore_start_sec = 0.0;

    let mut i = 0;
    while i + window_samples <= flow.samples.len() {
        let window = &flow.samples[i..i + window_samples];
        let t = i as f64 / sr;

        // Compute high-frequency energy ratio
        // 1. Remove low-frequency (breathing) component via simple differencing
        let diff: Vec<f64> = window.windows(2).map(|w| w[1] - w[0]).collect();

        // 2. Compute RMS of the differentiated signal
        let rms_diff = rms(&diff);

        // 3. Compute RMS of the original window
        let rms_orig = rms(window);

        // 4. High-frequency ratio — high ratio = more vibration = snoring
        let hf_ratio = if rms_orig > 0.1 { rms_diff / rms_orig } else { 0.0 };

        // 5. Zero-crossing rate — snoring has high ZCR
        let zcr = zero_crossing_rate(&diff);

        // Snore detection criteria
        let is_snoring = hf_ratio > 0.5 && zcr > 0.3 && rms_orig > 2.0;

        if is_snoring && !in_snore {
            in_snore = true;
            snore_start_sec = t;
        } else if !is_snoring && in_snore {
            in_snore = false;
            let end_sec = t;
            let duration = end_sec - snore_start_sec;

            if duration >= 1.0 {
                let mid_sec = (snore_start_sec + end_sec) / 2.0;
                let pos = position.value_at(mid_sec)
                    .map(BodyPosition::from_code)
                    .unwrap_or(BodyPosition::Unknown);

                events.push(SleepEvent::new(EventType::Snore, snore_start_sec, end_sec, pos));
            }
        }

        i += step;
    }

    events
}

fn rms(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
    (sum_sq / data.len() as f64).sqrt()
}

fn zero_crossing_rate(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let crossings = data.windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count();
    crossings as f64 / (data.len() - 1) as f64
}
