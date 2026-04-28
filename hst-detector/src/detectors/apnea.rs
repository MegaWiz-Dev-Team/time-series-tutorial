use crate::edf::signals::{Recording, BodyPosition};
use crate::events::{SleepEvent, EventType};
use crate::preprocess::CleanedData;

/// Minimum apnea duration in seconds (AASM rule)
const MIN_DURATION_SEC: f64 = 10.0;

/// Detect apnea events and classify as Obstructive / Central / Mixed
pub fn detect(recording: &Recording, cleaned: &CleanedData, flow_drop_threshold: f64) -> Vec<SleepEvent> {
    let mut events = Vec::new();
    let flow = &recording.flow;
    let effort = &recording.effort;
    let envelope = &cleaned.flow_envelope;
    let baseline = &cleaned.flow_baseline;
    let sr = flow.sample_rate;

    let min_samples = (MIN_DURATION_SEC * sr) as usize;

    // Scan for periods where envelope < threshold * baseline
    let mut in_apnea = false;
    let mut start_idx: usize = 0;

    for i in 0..envelope.len() {
        let threshold = baseline[i] * flow_drop_threshold;
        let is_low = envelope[i] < threshold && baseline[i] > 1.0; // baseline > 1.0 to avoid noise

        if is_low && !in_apnea {
            in_apnea = true;
            start_idx = i;
        } else if !is_low && in_apnea {
            in_apnea = false;
            let duration_samples = i - start_idx;

            if duration_samples >= min_samples {
                let start_sec = start_idx as f64 / sr;
                let end_sec = i as f64 / sr;
                let mid_sec = (start_sec + end_sec) / 2.0;

                // Get body position at event midpoint
                let position = recording.position.value_at(mid_sec)
                    .map(BodyPosition::from_code)
                    .unwrap_or(BodyPosition::Unknown);

                // Classify by effort signal variance
                let event_type = classify_apnea(effort, start_sec, end_sec);

                let mut event = SleepEvent::new(event_type, start_sec, end_sec, position);

                // Attach SpO2 nadir during/after the event
                let spo2_start = start_sec;
                let spo2_end = (end_sec + 30.0).min(recording.duration_sec);
                let spo2_slice = recording.spo2.slice_time(spo2_start, spo2_end);
                let valid_spo2: Vec<f64> = spo2_slice.iter()
                    .copied()
                    .filter(|&v| v <= 100.0 && v >= 50.0)
                    .collect();
                if let Some(&nadir) = valid_spo2.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    event.spo2_nadir = Some(nadir);
                }

                events.push(event);
            }
        }
    }

    events
}

/// Classify apnea as Obstructive, Central, or Mixed based on effort signal
fn classify_apnea(effort: &crate::edf::signals::Signal, start_sec: f64, end_sec: f64) -> EventType {
    let effort_slice = effort.slice_time(start_sec, end_sec);

    if effort_slice.is_empty() {
        return EventType::UnclassifiedApnea;
    }

    let duration = end_sec - start_sec;
    let mid_point = effort_slice.len() / 2;

    // Calculate variance for first half and second half
    let first_half = &effort_slice[..mid_point];
    let second_half = &effort_slice[mid_point..];

    let var_first = variance(first_half);
    let var_second = variance(second_half);
    let var_total = variance(effort_slice);

    // Thresholds for effort classification
    // Low variance = no respiratory effort (Central)
    // High variance = respiratory effort present (Obstructive)
    let low_threshold = 5.0; // empirical threshold for effort variance

    if var_total < low_threshold {
        // No effort throughout → Central
        EventType::CentralApnea
    } else if var_first < low_threshold && var_second >= low_threshold {
        // No effort first, then effort appears → Mixed
        EventType::MixedApnea
    } else if var_total >= low_threshold {
        // Effort present → Obstructive
        EventType::ObstructiveApnea
    } else {
        EventType::UnclassifiedApnea
    }
}

fn variance(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}
