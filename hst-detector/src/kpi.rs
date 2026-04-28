use std::collections::HashMap;
use serde::Serialize;

use crate::edf::signals::{Recording, BodyPosition};
use crate::events::{SleepEvent, EventType};
use crate::preprocess::CleanedData;

#[derive(Debug, Clone, Serialize)]
pub struct ClinicalReport {
    // Recording
    pub recording_duration_hours: f64,
    pub valid_duration_hours: f64,

    // Signal quality
    pub sqi_flow: f64,
    pub sqi_spo2: f64,
    pub sqi_pulse: f64,

    // Primary indices
    pub ahi: f64,
    pub ahi_supine: f64,
    pub ahi_non_supine: f64,
    pub odi: f64,

    // Oxygen metrics
    pub mean_spo2: f64,
    pub min_spo2: f64,
    pub t90: f64,  // % time SpO2 < 90%
    pub t88: f64,  // % time SpO2 < 88%

    // Event counts
    pub obstructive_apnea_count: u32,
    pub central_apnea_count: u32,
    pub mixed_apnea_count: u32,
    pub unclassified_apnea_count: u32,
    pub hypopnea_count: u32,
    pub desaturation_count: u32,
    pub csr_duration_min: f64,
    pub snore_count: u32,
    pub total_ahi_events: u32,

    // Position
    pub position_time_min: HashMap<String, f64>,

    // Severity
    pub osa_severity: String,
    pub positional_osa: bool,
}

#[derive(Debug, Serialize)]
pub struct FullResult {
    pub report: ClinicalReport,
    pub events: Vec<SleepEvent>,
}

/// Calculate all clinical KPIs from detected events
pub fn calculate(recording: &Recording, cleaned: &CleanedData, events: &[SleepEvent]) -> ClinicalReport {
    let total_hours = recording.duration_sec / 3600.0;
    let valid_hours = cleaned.valid_duration_sec / 3600.0;

    // Count events by type
    let obs = count_type(events, &EventType::ObstructiveApnea);
    let cen = count_type(events, &EventType::CentralApnea);
    let mix = count_type(events, &EventType::MixedApnea);
    let unc = count_type(events, &EventType::UnclassifiedApnea);
    let hyp = count_type(events, &EventType::Hypopnea);
    let des = count_type(events, &EventType::Desaturation);
    let snr = count_type(events, &EventType::Snore);

    let total_ahi_events = obs + cen + mix + unc + hyp;

    // AHI
    let ahi = if valid_hours > 0.0 { total_ahi_events as f64 / valid_hours } else { 0.0 };

    // AHI by position
    let supine_events = events.iter()
        .filter(|e| e.event_type.is_ahi_event() && e.position == BodyPosition::Supine)
        .count() as u32;
    let non_supine_events = total_ahi_events - supine_events;

    let supine_time = calculate_position_time(&recording.position, BodyPosition::Supine);
    let non_supine_time = recording.duration_sec - supine_time;

    let ahi_supine = if supine_time > 0.0 {
        supine_events as f64 / (supine_time / 3600.0)
    } else { 0.0 };

    let ahi_non_supine = if non_supine_time > 0.0 {
        non_supine_events as f64 / (non_supine_time / 3600.0)
    } else { 0.0 };

    // ODI
    let odi = if valid_hours > 0.0 { des as f64 / valid_hours } else { 0.0 };

    // SpO2 metrics
    let valid_spo2: Vec<f64> = recording.spo2.samples.iter()
        .copied()
        .filter(|&v| v <= 100.0 && v >= 50.0)
        .collect();

    let mean_spo2 = if !valid_spo2.is_empty() {
        valid_spo2.iter().sum::<f64>() / valid_spo2.len() as f64
    } else { 0.0 };

    let min_spo2 = valid_spo2.iter().cloned()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let total_valid = valid_spo2.len() as f64;
    let t90 = if total_valid > 0.0 {
        valid_spo2.iter().filter(|&&v| v < 90.0).count() as f64 / total_valid * 100.0
    } else { 0.0 };

    let t88 = if total_valid > 0.0 {
        valid_spo2.iter().filter(|&&v| v < 88.0).count() as f64 / total_valid * 100.0
    } else { 0.0 };

    // CSR duration
    let csr_duration_min: f64 = events.iter()
        .filter(|e| e.event_type == EventType::CheyneStokes)
        .map(|e| e.duration_sec / 60.0)
        .sum();

    // Position time breakdown
    let mut position_time_min = HashMap::new();
    for pos in &[BodyPosition::Supine, BodyPosition::Prone, BodyPosition::Left,
                 BodyPosition::Right, BodyPosition::Upright, BodyPosition::Unknown] {
        let time = calculate_position_time(&recording.position, *pos);
        if time > 0.0 {
            position_time_min.insert(pos.label().to_string(), time / 60.0);
        }
    }

    // OSA Severity
    let osa_severity = classify_severity(ahi);
    let positional_osa = ahi_non_supine > 0.0 && (ahi_supine / ahi_non_supine) > 2.0;

    ClinicalReport {
        recording_duration_hours: total_hours,
        valid_duration_hours: valid_hours,
        sqi_flow: cleaned.sqi_flow,
        sqi_spo2: cleaned.sqi_spo2,
        sqi_pulse: cleaned.sqi_pulse,
        ahi,
        ahi_supine,
        ahi_non_supine,
        odi,
        mean_spo2,
        min_spo2,
        t90,
        t88,
        obstructive_apnea_count: obs,
        central_apnea_count: cen,
        mixed_apnea_count: mix,
        unclassified_apnea_count: unc,
        hypopnea_count: hyp,
        desaturation_count: des,
        csr_duration_min,
        snore_count: snr,
        total_ahi_events,
        position_time_min,
        osa_severity: osa_severity.to_string(),
        positional_osa,
    }
}

fn count_type(events: &[SleepEvent], event_type: &EventType) -> u32 {
    events.iter().filter(|e| &e.event_type == event_type).count() as u32
}

fn calculate_position_time(position: &crate::edf::signals::Signal, target: BodyPosition) -> f64 {
    let count = position.samples.iter()
        .filter(|&&v| BodyPosition::from_code(v) == target)
        .count();
    count as f64 / position.sample_rate
}

fn classify_severity(ahi: f64) -> &'static str {
    if ahi < 5.0 { "Normal" }
    else if ahi < 15.0 { "Mild" }
    else if ahi < 30.0 { "Moderate" }
    else { "Severe" }
}

/// Format the clinical report as human-readable text
pub fn format_text_report(report: &ClinicalReport) -> String {
    let mut s = String::new();
    s.push_str("╔══════════════════════════════════════════════════════════════╗\n");
    s.push_str("║         🛏️  HST Sleep Event Detection Report               ║\n");
    s.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

    s.push_str("📋 Recording Info\n");
    s.push_str(&format!("   Duration:       {:.1} hours\n", report.recording_duration_hours));
    s.push_str(&format!("   Valid time:     {:.1} hours\n", report.valid_duration_hours));
    s.push_str(&format!("   SQI Flow:       {:.0}%  |  SpO2: {:.0}%  |  Pulse: {:.0}%\n\n",
        report.sqi_flow * 100.0, report.sqi_spo2 * 100.0, report.sqi_pulse * 100.0));

    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str("📊 PRIMARY INDICES\n");
    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str(&format!("   AHI:            {:.1} events/hr   → {}\n", report.ahi, report.osa_severity));
    s.push_str(&format!("   AHI (Supine):   {:.1} events/hr\n", report.ahi_supine));
    s.push_str(&format!("   AHI (Non-sup):  {:.1} events/hr\n", report.ahi_non_supine));
    s.push_str(&format!("   Positional OSA: {}\n", if report.positional_osa { "Yes" } else { "No" }));
    s.push_str(&format!("   ODI:            {:.1} events/hr\n\n", report.odi));

    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str("🫁 OXYGEN METRICS\n");
    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str(&format!("   Mean SpO2:      {:.1}%\n", report.mean_spo2));
    s.push_str(&format!("   Nadir SpO2:     {:.0}%\n", report.min_spo2));
    s.push_str(&format!("   T90:            {:.1}% of time\n", report.t90));
    s.push_str(&format!("   T88:            {:.1}% of time\n\n", report.t88));

    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str("📈 EVENT COUNTS\n");
    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str(&format!("   🔴 Obstructive Apnea:   {}\n", report.obstructive_apnea_count));
    s.push_str(&format!("   🔵 Central Apnea:       {}\n", report.central_apnea_count));
    s.push_str(&format!("   🟣 Mixed Apnea:         {}\n", report.mixed_apnea_count));
    s.push_str(&format!("   🩷 Unclassified Apnea:  {}\n", report.unclassified_apnea_count));
    s.push_str(&format!("   🟢 Hypopnea:            {}\n", report.hypopnea_count));
    s.push_str(&format!("   🔵 Desaturation:        {}\n", report.desaturation_count));
    s.push_str(&format!("   🟠 Snore:               {}\n", report.snore_count));
    s.push_str(&format!("   ─── Total AHI events:   {}\n", report.total_ahi_events));
    s.push_str(&format!("   🔵 CSR duration:        {:.1} min\n\n", report.csr_duration_min));

    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    s.push_str("🛏️ BODY POSITION (minutes)\n");
    s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let mut positions: Vec<_> = report.position_time_min.iter().collect();
    positions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (pos, mins) in positions {
        let pct = mins / (report.recording_duration_hours * 60.0) * 100.0;
        s.push_str(&format!("   {:10} {:6.1} min  ({:.0}%)\n", pos, mins, pct));
    }

    s.push('\n');
    s.push_str("══════════════════════════════════════════════════════════════\n");

    s
}
