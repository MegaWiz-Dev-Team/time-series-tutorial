use crate::edf::signals::{Recording, Signal};
use std::io::Read;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("EDF parse error: {0}")]
    Edf(String),
    #[error("Missing signal: {0}")]
    MissingSignal(String),
}

/// Load either an .mmrx ZIP archive or a direct .edf file
pub fn load_file(path: &Path) -> Result<Recording, ReaderError> {
    if path.extension().and_then(|s| s.to_str()) == Some("edf") {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        parse_edf_bytes(&bytes)
    } else {
        load_mmrx(path)
    }
}

/// Load an .mmrx file (ZIP archive containing EDF+ files)
pub fn load_mmrx(path: &Path) -> Result<Recording, ReaderError> {
    // Step 1: Open ZIP archive
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Step 2: Find the main EDF file (largest .edf, not *_Events.edf)
    let edf_name = find_main_edf(&mut archive)?;

    // Step 3: Extract EDF bytes
    let mut edf_file = archive.by_name(&edf_name)?;
    let mut edf_bytes = Vec::new();
    edf_file.read_to_end(&mut edf_bytes)?;
    drop(edf_file);

    // Step 4: Parse EDF+ manually (for maximum compatibility with ApneaLink Air)
    parse_edf_bytes(&edf_bytes)
}

fn find_main_edf(archive: &mut zip::ZipArchive<std::fs::File>) -> Result<String, ReaderError> {
    let mut best_name = None;
    let mut best_size = 0u64;

    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if name.ends_with(".edf") && !name.contains("Events") && entry.size() > best_size {
            best_size = entry.size();
            best_name = Some(name);
        }
    }

    best_name.ok_or_else(|| ReaderError::Edf("No main EDF file found in archive".into()))
}

/// Parse EDF+ bytes according to the EDF specification
/// Reference: https://www.edfplus.info/specs/edf.html
fn parse_edf_bytes(bytes: &[u8]) -> Result<Recording, ReaderError> {
    if bytes.len() < 256 {
        return Err(ReaderError::Edf("File too small for EDF header".into()));
    }

    // --- Fixed header (256 bytes) ---
    let _version = read_ascii(bytes, 0, 8);
    let _patient = read_ascii(bytes, 8, 80);
    let _recording = read_ascii(bytes, 88, 80);
    let start_date = read_ascii(bytes, 168, 8);     // dd.mm.yy
    let start_time = read_ascii(bytes, 176, 8);     // hh.mm.ss
    let header_bytes: usize = read_ascii(bytes, 184, 8).trim().parse()
        .map_err(|_| ReaderError::Edf("Invalid header size".into()))?;
    let _subtype = read_ascii(bytes, 192, 44);       // EDF+C or EDF+D
    let num_records: usize = read_ascii(bytes, 236, 8).trim().parse()
        .map_err(|_| ReaderError::Edf("Invalid number of records".into()))?;
    let record_duration: f64 = read_ascii(bytes, 244, 8).trim().parse()
        .map_err(|_| ReaderError::Edf("Invalid record duration".into()))?;
    let num_signals: usize = read_ascii(bytes, 252, 4).trim().parse()
        .map_err(|_| ReaderError::Edf("Invalid number of signals".into()))?;

    // --- Signal headers (256 bytes per signal) ---
    let sh_offset = 256; // start of signal headers
    let labels: Vec<String> = (0..num_signals)
        .map(|i| read_ascii(bytes, sh_offset + i * 16, 16).trim().to_string())
        .collect();
    
    // Skip transducer type (80 * ns), physical dimension, physical min/max, digital min/max
    let phys_dim_offset = sh_offset + num_signals * 96;
    let phys_dims: Vec<String> = (0..num_signals)
        .map(|i| read_ascii(bytes, phys_dim_offset + i * 8, 8).trim().to_string())
        .collect();

    let phys_min_offset = sh_offset + num_signals * 104;
    let phys_mins: Vec<f64> = (0..num_signals)
        .map(|i| read_ascii(bytes, phys_min_offset + i * 8, 8).trim().parse().unwrap_or(0.0))
        .collect();

    let phys_max_offset = sh_offset + num_signals * 112;
    let phys_maxs: Vec<f64> = (0..num_signals)
        .map(|i| read_ascii(bytes, phys_max_offset + i * 8, 8).trim().parse().unwrap_or(1.0))
        .collect();

    let dig_min_offset = sh_offset + num_signals * 120;
    let dig_mins: Vec<f64> = (0..num_signals)
        .map(|i| read_ascii(bytes, dig_min_offset + i * 8, 8).trim().parse().unwrap_or(-32768.0))
        .collect();

    let dig_max_offset = sh_offset + num_signals * 128;
    let dig_maxs: Vec<f64> = (0..num_signals)
        .map(|i| read_ascii(bytes, dig_max_offset + i * 8, 8).trim().parse().unwrap_or(32767.0))
        .collect();

    // Skip prefilter (80 * ns)
    let samples_per_record_offset = sh_offset + num_signals * 216;
    let samples_per_record: Vec<usize> = (0..num_signals)
        .map(|i| {
            read_ascii(bytes, samples_per_record_offset + i * 8, 8)
                .trim().parse().unwrap_or(0)
        })
        .collect();

    // --- Parse start datetime ---
    let start_datetime = parse_edf_datetime(&start_date, &start_time);

    // --- Read data records ---
    let data_offset = header_bytes;
    let mut all_samples: Vec<Vec<f64>> = vec![Vec::new(); num_signals];

    let record_size_samples: usize = samples_per_record.iter().sum();
    let record_size_bytes = record_size_samples * 2; // 16-bit integers

    for rec_idx in 0..num_records {
        let rec_start = data_offset + rec_idx * record_size_bytes;
        if rec_start + record_size_bytes > bytes.len() {
            break; // truncated file
        }

        let mut offset = rec_start;
        for sig_idx in 0..num_signals {
            let n = samples_per_record[sig_idx];
            for _ in 0..n {
                if offset + 2 > bytes.len() { break; }
                let raw = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                // Convert digital to physical
                let physical = digital_to_physical(
                    raw as f64,
                    dig_mins[sig_idx], dig_maxs[sig_idx],
                    phys_mins[sig_idx], phys_maxs[sig_idx],
                );
                all_samples[sig_idx].push(physical);
                offset += 2;
            }
        }
    }

    // --- Build Recording struct ---
    let duration_sec = num_records as f64 * record_duration;

    let make_signal = |idx: usize| -> Signal {
        let sr = samples_per_record[idx] as f64 / record_duration;
        Signal {
            label: labels[idx].clone(),
            samples: all_samples[idx].clone(),
            sample_rate: sr,
            unit: phys_dims[idx].clone(),
        }
    };

    // Find signal indices by label
    let find_idx = |name: &str| -> Result<usize, ReaderError> {
        labels.iter().position(|l| l.starts_with(name))
            .ok_or_else(|| ReaderError::MissingSignal(name.to_string()))
    };

    let flow_idx = find_idx("Resp nasal")?;
    let effort_idx = find_idx("Resp thorax")?;
    let pulse_idx = find_idx("Pulse")?;
    let spo2_idx = find_idx("SaO2")?;
    let pos_idx = find_idx("Position")?;
    let acc_x_idx = find_idx("Acc x")?;
    let acc_y_idx = find_idx("Acc y")?;
    let acc_z_idx = find_idx("Acc z")?;

    Ok(Recording {
        start_time: start_datetime,
        duration_sec,
        flow: make_signal(flow_idx),
        effort: make_signal(effort_idx),
        pulse: make_signal(pulse_idx),
        spo2: make_signal(spo2_idx),
        position: make_signal(pos_idx),
        accel_x: make_signal(acc_x_idx),
        accel_y: make_signal(acc_y_idx),
        accel_z: make_signal(acc_z_idx),
    })
}

fn digital_to_physical(raw: f64, dig_min: f64, dig_max: f64, phys_min: f64, phys_max: f64) -> f64 {
    let scale = (phys_max - phys_min) / (dig_max - dig_min);
    phys_min + (raw - dig_min) * scale
}

fn read_ascii(bytes: &[u8], offset: usize, len: usize) -> String {
    let end = (offset + len).min(bytes.len());
    String::from_utf8_lossy(&bytes[offset..end]).to_string()
}

fn parse_edf_datetime(date_str: &str, time_str: &str) -> chrono::NaiveDateTime {
    // Format: dd.mm.yy and hh.mm.ss
    let parts_d: Vec<&str> = date_str.trim().split('.').collect();
    let parts_t: Vec<&str> = time_str.trim().split('.').collect();

    let day: u32 = parts_d.first().and_then(|s| s.parse().ok()).unwrap_or(1);
    let month: u32 = parts_d.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let year_short: i32 = parts_d.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let year = if year_short >= 85 { 1900 + year_short } else { 2000 + year_short };

    let hour: u32 = parts_t.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let min: u32 = parts_t.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let sec: u32 = parts_t.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    chrono::NaiveDate::from_ymd_opt(year, month, day)
        .unwrap_or(chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
        .and_hms_opt(hour, min, sec)
        .unwrap_or_default()
}
