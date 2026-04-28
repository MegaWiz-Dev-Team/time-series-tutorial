pub mod apnea;
pub mod hypopnea;
pub mod desaturation;
pub mod cheyne_stokes;
pub mod snore;
pub mod position;

use crate::edf::signals::Recording;
use crate::events::SleepEvent;
use crate::preprocess::CleanedData;

/// Run all detectors and return combined event list, sorted by onset time
pub fn detect_all(recording: &Recording, cleaned: &CleanedData, verbose: bool, apnea_threshold: f64, hypopnea_threshold: f64) -> Vec<SleepEvent> {
    let mut all_events = Vec::new();

    // Excluded data
    let excluded = position::detect_excluded(cleaned);
    if verbose { eprintln!("   🔘 Excluded segments: {}", excluded.len()); }
    all_events.extend(excluded);

    // Desaturations (run first — needed by hypopnea detector)
    let desats = desaturation::detect(&recording.spo2, &cleaned.spo2_baseline, &recording.position);
    if verbose { eprintln!("   🔵 Desaturations: {}", desats.len()); }
    all_events.extend(desats.clone());

    // Apneas (Obstructive / Central / Mixed)
    let apneas = apnea::detect(recording, cleaned, apnea_threshold);
    if verbose {
        let obs = apneas.iter().filter(|e| e.event_type == crate::events::EventType::ObstructiveApnea).count();
        let cen = apneas.iter().filter(|e| e.event_type == crate::events::EventType::CentralApnea).count();
        let mix = apneas.iter().filter(|e| e.event_type == crate::events::EventType::MixedApnea).count();
        let unc = apneas.iter().filter(|e| e.event_type == crate::events::EventType::UnclassifiedApnea).count();
        eprintln!("   🔴 Apneas: {} (Obs:{}, Cen:{}, Mix:{}, Unc:{})",
            apneas.len(), obs, cen, mix, unc);
    }
    all_events.extend(apneas);

    // Hypopneas
    let hypops = hypopnea::detect(recording, cleaned, &desats, hypopnea_threshold, apnea_threshold);
    if verbose { eprintln!("   🟢 Hypopneas: {}", hypops.len()); }
    all_events.extend(hypops);

    // Cheyne-Stokes
    let csr = cheyne_stokes::detect(recording, cleaned);
    if verbose { eprintln!("   🔵 Cheyne-Stokes segments: {}", csr.len()); }
    all_events.extend(csr);

    // Snoring
    let snores = snore::detect(&recording.flow, &recording.position);
    if verbose { eprintln!("   🟠 Snore events: {}", snores.len()); }
    all_events.extend(snores);

    // Sort all events by onset time
    all_events.sort_by(|a, b| a.start_sec.partial_cmp(&b.start_sec).unwrap());

    all_events
}
