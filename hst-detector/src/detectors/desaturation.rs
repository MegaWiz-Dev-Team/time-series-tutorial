use crate::edf::signals::{Signal, BodyPosition};
use crate::events::{SleepEvent, EventType};

const SPO2_DROP_THRESHOLD: f64 = 3.0;
const MIN_DESAT_DURATION_SEC: f64 = 2.0;

/// Detect oxygen desaturation events (SpO2 drop >= 3% from baseline)
pub fn detect(spo2: &Signal, spo2_baseline: &[f64], position: &Signal) -> Vec<SleepEvent> {
    let mut events = Vec::new();
    let sr = spo2.sample_rate;

    let mut in_desat = false;
    let mut start_idx: usize = 0;
    let mut nadir = f64::MAX;

    for i in 0..spo2.samples.len() {
        let val = spo2.samples[i];
        if val > 100.0 || val < 50.0 { continue; } // skip sentinel/invalid

        let baseline_val = if i < spo2_baseline.len() { spo2_baseline[i] } else { 95.0 };
        let drop = baseline_val - val;

        if drop >= SPO2_DROP_THRESHOLD && !in_desat {
            in_desat = true;
            start_idx = i;
            nadir = val;
        } else if drop >= SPO2_DROP_THRESHOLD && in_desat {
            nadir = nadir.min(val);
        } else if drop < SPO2_DROP_THRESHOLD && in_desat {
            in_desat = false;
            let start_sec = start_idx as f64 / sr;
            let end_sec = i as f64 / sr;
            let duration = end_sec - start_sec;

            if duration >= MIN_DESAT_DURATION_SEC {
                let mid_sec = (start_sec + end_sec) / 2.0;
                let pos = position.value_at(mid_sec)
                    .map(BodyPosition::from_code)
                    .unwrap_or(BodyPosition::Unknown);

                let baseline_at_start = if start_idx < spo2_baseline.len() {
                    spo2_baseline[start_idx]
                } else {
                    95.0
                };

                let mut event = SleepEvent::new(EventType::Desaturation, start_sec, end_sec, pos);
                event.spo2_nadir = Some(nadir);
                event.spo2_drop = Some(baseline_at_start - nadir);
                events.push(event);
            }
            nadir = f64::MAX;
        }
    }

    events
}
