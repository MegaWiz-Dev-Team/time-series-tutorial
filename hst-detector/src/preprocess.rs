use crate::edf::signals::{Recording, Signal, BodyPosition};
use crate::events::{SleepEvent, EventType};

/// Cleaned / preprocessed data ready for event detection
pub struct CleanedData {
    /// Flow baseline (95th percentile, 2-min sliding window)
    pub flow_baseline: Vec<f64>,
    /// Flow envelope (rectified + smoothed)
    pub flow_envelope: Vec<f64>,
    /// SpO2 baseline (moving average, 2-min window)
    pub spo2_baseline: Vec<f64>,
    /// Segments of excluded/invalid data
    pub excluded_segments: Vec<(f64, f64)>, // (start_sec, end_sec)
    /// Signal quality index per channel (0.0 - 1.0)
    pub sqi_flow: f64,
    pub sqi_spo2: f64,
    pub sqi_pulse: f64,
    /// Valid recording duration in seconds (excluding excluded segments)
    pub valid_duration_sec: f64,
}

/// Clean all signals and compute baselines
pub fn clean_recording(recording: &Recording, verbose: bool) -> CleanedData {
    if verbose {
        eprintln!("🧹 Preprocessing signals...");
    }

    // 1. Detect excluded segments from sentinel values
    let excluded_spo2 = detect_sentinels(&recording.spo2, 127.0);
    let excluded_pulse = detect_sentinels(&recording.pulse, 511.0);

    // Merge excluded segments
    let mut excluded_segments = Vec::new();
    excluded_segments.extend_from_slice(&excluded_spo2);
    excluded_segments.extend_from_slice(&excluded_pulse);
    excluded_segments = merge_segments(excluded_segments);

    let valid_duration_sec = recording.duration_sec
        - excluded_segments.iter().map(|(s, e)| e - s).sum::<f64>();

    // 2. Signal quality indices
    let sqi_flow = 1.0; // Flow doesn't have sentinel values
    let sqi_spo2 = compute_sqi(&recording.spo2, 127.0);
    let sqi_pulse = compute_sqi(&recording.pulse, 511.0);

    if verbose {
        eprintln!("   SQI — Flow: {:.1}%, SpO2: {:.1}%, Pulse: {:.1}%",
            sqi_flow * 100.0, sqi_spo2 * 100.0, sqi_pulse * 100.0);
        eprintln!("   Valid duration: {:.1} hours ({:.0} sec excluded)",
            valid_duration_sec / 3600.0,
            recording.duration_sec - valid_duration_sec);
    }

    // 3. Compute flow envelope (rectified + smoothed)
    let flow_envelope = compute_envelope(&recording.flow);

    // 4. Compute flow baseline (95th percentile over 2-min window)
    let flow_baseline = compute_percentile_baseline(&flow_envelope, recording.flow.sample_rate, 120.0, 0.95);

    // 5. Compute SpO2 baseline (moving average over 2-min window, ignoring sentinels)
    let spo2_baseline = compute_spo2_baseline(&recording.spo2, 120.0);

    if verbose {
        eprintln!("   ✅ Preprocessing complete");
    }

    CleanedData {
        flow_baseline,
        flow_envelope,
        spo2_baseline,
        excluded_segments,
        sqi_flow,
        sqi_spo2,
        sqi_pulse,
        valid_duration_sec,
    }
}

/// Detect segments where signal equals a sentinel value
fn detect_sentinels(signal: &Signal, sentinel: f64) -> Vec<(f64, f64)> {
    let mut segments = Vec::new();
    let mut in_sentinel = false;
    let mut start = 0.0;

    for (i, &val) in signal.samples.iter().enumerate() {
        let t = i as f64 / signal.sample_rate;
        if (val - sentinel).abs() < 0.5 {
            if !in_sentinel {
                in_sentinel = true;
                start = t;
            }
        } else if in_sentinel {
            in_sentinel = false;
            segments.push((start, t));
        }
    }
    if in_sentinel {
        segments.push((start, signal.duration_sec()));
    }
    segments
}

/// Signal Quality Index: fraction of time the signal is NOT sentinel
fn compute_sqi(signal: &Signal, sentinel: f64) -> f64 {
    let valid = signal.samples.iter().filter(|&&v| (v - sentinel).abs() >= 0.5).count();
    valid as f64 / signal.samples.len() as f64
}

/// Compute signal envelope using rectification + smoothing
pub fn compute_envelope(flow: &Signal) -> Vec<f64> {
    let samples = &flow.samples;
    let n = samples.len();
    let mut envelope = vec![0.0; n];

    // Rectify (absolute value)
    let rectified: Vec<f64> = samples.iter().map(|&x| x.abs()).collect();

    // Smooth with a moving average (window = 1 second)
    let window = flow.sample_rate as usize;
    let half_w = window / 2;

    let mut running_sum: f64 = rectified[..window.min(n)].iter().sum();
    let mut count = window.min(n);

    for i in 0..n {
        let center = i;
        // Simple moving average centered on i
        let left = if center >= half_w { center - half_w } else { 0 };
        let right = (center + half_w).min(n - 1);
        let w = right - left + 1;

        let sum: f64 = rectified[left..=right].iter().sum();
        envelope[i] = sum / w as f64;
    }

    envelope
}

/// Compute a percentile-based baseline using sliding window
/// Optimized: downsample to 1 Hz first, compute percentile, then interpolate back
fn compute_percentile_baseline(data: &[f64], sample_rate: f64, window_sec: f64, percentile: f64) -> Vec<f64> {
    let n = data.len();

    // Step 1: Downsample to 1 Hz (take max per second for envelope baseline)
    let step = sample_rate as usize;
    let ds_len = n / step + 1;
    let mut downsampled = Vec::with_capacity(ds_len);
    for chunk_start in (0..n).step_by(step) {
        let chunk_end = (chunk_start + step).min(n);
        let max_val = data[chunk_start..chunk_end]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        downsampled.push(max_val);
    }

    // Step 2: Compute percentile on the 1 Hz downsampled data (much smaller window)
    let ds_window = window_sec as usize; // window in seconds = window in samples at 1 Hz
    let ds_half_w = ds_window / 2;
    let mut ds_baseline = vec![0.0; downsampled.len()];

    for i in 0..downsampled.len() {
        let left = i.saturating_sub(ds_half_w);
        let right = (i + ds_half_w).min(downsampled.len() - 1);

        let mut window_data: Vec<f64> = downsampled[left..=right].to_vec();
        window_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((window_data.len() as f64 * percentile) as usize).min(window_data.len() - 1);
        ds_baseline[i] = window_data[idx];
    }

    // Step 3: Interpolate back to full sample rate
    let mut baseline = vec![0.0; n];
    for i in 0..n {
        let ds_idx = i / step;
        let ds_next = (ds_idx + 1).min(downsampled.len() - 1);
        let frac = (i % step) as f64 / step as f64;
        baseline[i] = ds_baseline[ds_idx] * (1.0 - frac) + ds_baseline[ds_next] * frac;
    }

    baseline
}

/// Compute SpO2 moving average baseline, ignoring sentinel values (127)
fn compute_spo2_baseline(spo2: &Signal, window_sec: f64) -> Vec<f64> {
    let n = spo2.samples.len();
    let window = (window_sec * spo2.sample_rate) as usize;
    let half_w = window / 2;
    let mut baseline = vec![0.0; n];

    for i in 0..n {
        let left = if i >= half_w { i - half_w } else { 0 };
        let right = (i + half_w).min(n - 1);

        let valid: Vec<f64> = spo2.samples[left..=right]
            .iter()
            .copied()
            .filter(|&v| v <= 100.0 && v >= 50.0)
            .collect();

        baseline[i] = if valid.is_empty() {
            95.0 // default fallback
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        };
    }

    baseline
}

/// Merge overlapping time segments
fn merge_segments(mut segments: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if segments.is_empty() {
        return segments;
    }
    segments.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut merged = vec![segments[0]];
    for seg in segments.into_iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if seg.0 <= last.1 {
            last.1 = last.1.max(seg.1);
        } else {
            merged.push(seg);
        }
    }
    merged
}
