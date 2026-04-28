use crate::edf::signals::BodyPosition;
use crate::events::{SleepEvent, EventType};
use crate::preprocess::CleanedData;

/// Convert excluded data segments into SleepEvents
pub fn detect_excluded(cleaned: &CleanedData) -> Vec<SleepEvent> {
    cleaned.excluded_segments.iter().map(|&(start, end)| {
        SleepEvent::new(EventType::ExcludedData, start, end, BodyPosition::Unknown)
    }).collect()
}
