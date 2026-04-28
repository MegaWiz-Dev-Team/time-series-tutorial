use crate::edf::signals::{Recording, BodyPosition};
use crate::events::{SleepEvent, EventType};
use crate::preprocess::CleanedData;

/// AASM V3: Flow reduction >= 30% (but < 90%) for >= 10 seconds
/// with >= 3% SpO2 desaturation
const MIN_DURATION_SEC: f64 = 10.0;
const FLOW_DROP_MAX: f64 = 0.90; // < 90% reduction (otherwise it's apnea)
const SPO2_DROP_THRESHOLD: f64 = 3.0;

/// Detect hypopnea events following AASM V3 criteria
pub fn detect(
    recording: &Recording,
    cleaned: &CleanedData,
    desats: &[SleepEvent],
    flow_drop_min: f64,
    apnea_threshold: f64,
) -> Vec<SleepEvent> {
    let mut events = Vec::new();
    let envelope = &cleaned.flow_envelope;
    let baseline = &cleaned.flow_baseline;
    let sr = recording.flow.sample_rate;
    let min_samples = (MIN_DURATION_SEC * sr) as usize;

    let mut in_hypopnea = false;
    let mut start_idx: usize = 0;

    for i in 0..envelope.len() {
        if baseline[i] < 1.0 { continue; } // skip if baseline too low (noise)

        let ratio = envelope[i] / baseline[i];
        let is_reduced = ratio <= (1.0 - flow_drop_min) && ratio > apnea_threshold;

        if is_reduced && !in_hypopnea {
            in_hypopnea = true;
            start_idx = i;
        } else if !is_reduced && in_hypopnea {
            in_hypopnea = false;
            let duration_samples = i - start_idx;

            if duration_samples >= min_samples {
                let start_sec = start_idx as f64 / sr;
                let end_sec = i as f64 / sr;
                let mid_sec = (start_sec + end_sec) / 2.0;

                // Check for associated SpO2 desaturation (within event or up to 30s after)
                let has_desat = check_desaturation(
                    &recording.spo2,
                    &cleaned.spo2_baseline,
                    start_sec,
                    end_sec + 30.0,
                    SPO2_DROP_THRESHOLD,
                ) || has_overlapping_desat(desats, start_sec, end_sec + 30.0);

                if has_desat {
                    let position = recording.position.value_at(mid_sec)
                        .map(BodyPosition::from_code)
                        .unwrap_or(BodyPosition::Unknown);

                    let mut event = SleepEvent::new(EventType::Hypopnea, start_sec, end_sec, position);

                    // Record SpO2 nadir
                    let spo2_window = recording.spo2.slice_time(start_sec, (end_sec + 30.0).min(recording.duration_sec));
                    let valid: Vec<f64> = spo2_window.iter()
                        .copied()
                        .filter(|&v| v <= 100.0 && v >= 50.0)
                        .collect();
                    if let Some(&nadir) = valid.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                        event.spo2_nadir = Some(nadir);
                    }

                    events.push(event);
                }
            }
        }
    }

    events
}

/// Check if SpO2 drops >= threshold from baseline in a time window
fn check_desaturation(
    spo2: &crate::edf::signals::Signal,
    spo2_baseline: &[f64],
    start_sec: f64,
    end_sec: f64,
    threshold: f64,
) -> bool {
    let start_idx = (start_sec * spo2.sample_rate) as usize;
    let end_idx = ((end_sec * spo2.sample_rate) as usize).min(spo2.samples.len());

    for i in start_idx..end_idx {
        let val = spo2.samples[i];
        if val > 100.0 || val < 50.0 { continue; } // skip sentinels
        if i < spo2_baseline.len() {
            let drop = spo2_baseline[i] - val;
            if drop >= threshold {
                return true;
            }
        }
    }
    false
}

/// Check if any pre-detected desaturation event overlaps with a time window
fn has_overlapping_desat(desats: &[SleepEvent], start_sec: f64, end_sec: f64) -> bool {
    desats.iter().any(|d| d.start_sec < end_sec && d.end_sec > start_sec)
}
