use crate::edf::signals::{Recording, BodyPosition};
use crate::events::{SleepEvent, EventType};
use crate::preprocess::CleanedData;

/// AASM criteria for Cheyne-Stokes:
/// - At least 3 consecutive crescendo-decrescendo cycles
/// - Cycle length 40-90 seconds
/// - Central apneas/hypopneas at the troughs
const MIN_CYCLES: usize = 3;
const MIN_CYCLE_SEC: f64 = 40.0;
const MAX_CYCLE_SEC: f64 = 90.0;
const ANALYSIS_WINDOW_SEC: f64 = 300.0; // 5-min sliding window

/// Detect Cheyne-Stokes respiration patterns
pub fn detect(recording: &Recording, cleaned: &CleanedData) -> Vec<SleepEvent> {
    let mut events = Vec::new();
    let envelope = &cleaned.flow_envelope;
    let sr = recording.flow.sample_rate;
    let n = envelope.len();

    // Create a very smooth envelope for CSR detection (30s window)
    let smooth_window = (30.0 * sr) as usize;
    let smooth_env = smooth(envelope, smooth_window);

    // Find peaks and troughs in the smooth envelope
    let peaks = find_peaks(&smooth_env, sr, 20.0); // min 20s between peaks
    let troughs = find_troughs(&smooth_env, sr, 20.0);

    if peaks.len() < MIN_CYCLES || troughs.len() < MIN_CYCLES {
        return events;
    }

    // Look for sequences of regular peaks with appropriate cycle length
    for window_start in (0..peaks.len().saturating_sub(MIN_CYCLES)).step_by(1) {
        let window_peaks = &peaks[window_start..];
        if window_peaks.len() < MIN_CYCLES { break; }

        // Check cycle lengths between consecutive peaks
        let mut valid_cycles = 0;
        let mut last_valid_peak = window_start;

        for i in 0..window_peaks.len() - 1 {
            let cycle_samples = window_peaks[i + 1] - window_peaks[i];
            let cycle_sec = cycle_samples as f64 / sr;

            if cycle_sec >= MIN_CYCLE_SEC && cycle_sec <= MAX_CYCLE_SEC {
                valid_cycles += 1;
                last_valid_peak = window_start + i + 1;
            } else {
                break; // non-consecutive → restart
            }
        }

        if valid_cycles >= MIN_CYCLES {
            let start_sec = window_peaks[0] as f64 / sr;
            let end_sec = peaks[last_valid_peak] as f64 / sr;
            let mid_sec = (start_sec + end_sec) / 2.0;

            let pos = recording.position.value_at(mid_sec)
                .map(BodyPosition::from_code)
                .unwrap_or(BodyPosition::Unknown);

            events.push(SleepEvent::new(EventType::CheyneStokes, start_sec, end_sec, pos));
        }
    }

    // Merge overlapping CSR segments
    merge_csr_events(&mut events);

    events
}

fn smooth(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let half_w = window / 2;
    let mut result = vec![0.0; n];

    for i in 0..n {
        let left = i.saturating_sub(half_w);
        let right = (i + half_w).min(n - 1);
        let sum: f64 = data[left..=right].iter().sum();
        result[i] = sum / (right - left + 1) as f64;
    }
    result
}

fn find_peaks(data: &[f64], sr: f64, min_distance_sec: f64) -> Vec<usize> {
    let min_dist = (min_distance_sec * sr) as usize;
    let mut peaks = Vec::new();

    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            if peaks.last().map_or(true, |&last: &usize| i - last >= min_dist) {
                peaks.push(i);
            }
        }
    }
    peaks
}

fn find_troughs(data: &[f64], sr: f64, min_distance_sec: f64) -> Vec<usize> {
    let min_dist = (min_distance_sec * sr) as usize;
    let mut troughs = Vec::new();

    for i in 1..data.len() - 1 {
        if data[i] < data[i - 1] && data[i] < data[i + 1] {
            if troughs.last().map_or(true, |&last: &usize| i - last >= min_dist) {
                troughs.push(i);
            }
        }
    }
    troughs
}

fn merge_csr_events(events: &mut Vec<SleepEvent>) {
    if events.len() < 2 { return; }
    events.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());
    
    let mut merged = vec![events[0].clone()];
    for e in events.iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if e.start_sec <= last.end_sec {
            last.end_sec = last.end_sec.max(e.end_sec);
            last.duration_sec = last.end_sec - last.start_sec;
        } else {
            merged.push(e.clone());
        }
    }
    *events = merged;
}
