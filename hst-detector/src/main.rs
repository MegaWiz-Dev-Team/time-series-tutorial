mod edf;
mod events;
mod preprocess;
mod detectors;
mod kpi;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "hst-detector")]
#[command(about = "🛏️ Automated Sleep Event Detector for HST (Home Sleep Test) data")]
#[command(version)]
struct Cli {
    /// Path to .mmrx file (HST raw data)
    input: PathBuf,

    /// Output format: json or text
    #[arg(short, long, default_value = "text")]
    output: String,

    /// Output file path (default: stdout)
    #[arg(short = 'f', long)]
    output_file: Option<PathBuf>,

    /// Flow drop threshold for Apnea (0.0 to 1.0, e.g., 0.1 for >= 90% drop)
    #[arg(long, default_value_t = 0.1)]
    apnea_threshold: f64,

    /// Flow drop threshold for Hypopnea (0.0 to 1.0, e.g., 0.3 for >= 30% drop)
    #[arg(long, default_value_t = 0.3)]
    hypopnea_threshold: f64,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        eprintln!("🛏️  HST Sleep Event Detector v{}", env!("CARGO_PKG_VERSION"));
        eprintln!("📂 Input: {}", cli.input.display());
    }

    // Step 1: Extract .mmrx and read EDF
    let recording = match edf::reader::load_file(&cli.input) {
        Ok(rec) => {
            if cli.verbose {
                eprintln!("✅ Loaded recording: {:.1} hours, {} channels",
                    rec.duration_sec / 3600.0, rec.channel_count());
            }
            rec
        }
        Err(e) => {
            eprintln!("❌ Failed to load file: {}", e);
            std::process::exit(1);
        }
    };

    // Step 2: Preprocess signals
    let cleaned = preprocess::clean_recording(&recording, cli.verbose);

    // Step 3: Detect events
    let all_events = detectors::detect_all(&recording, &cleaned, cli.verbose, cli.apnea_threshold, cli.hypopnea_threshold);

    if cli.verbose {
        eprintln!("📊 Detected {} total events", all_events.len());
    }

    // Step 4: Calculate clinical KPIs
    let report = kpi::calculate(&recording, &cleaned, &all_events);

    // Step 5: Output results
    let output_str = match cli.output.as_str() {
        "json" => {
            let result = kpi::FullResult {
                report: report.clone(),
                events: all_events,
            };
            serde_json::to_string_pretty(&result).unwrap()
        }
        _ => kpi::format_text_report(&report),
    };

    match cli.output_file {
        Some(path) => {
            std::fs::write(&path, &output_str).expect("Failed to write output file");
            if cli.verbose {
                eprintln!("💾 Output written to {}", path.display());
            }
        }
        None => println!("{}", output_str),
    }
}
