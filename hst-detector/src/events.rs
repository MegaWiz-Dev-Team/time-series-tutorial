use serde::Serialize;

use crate::edf::signals::BodyPosition;

/// All possible sleep event types matching AirView labels
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum EventType {
    ObstructiveApnea,
    CentralApnea,
    MixedApnea,
    UnclassifiedApnea,
    Hypopnea,
    Desaturation,
    CheyneStokes,
    Snore,
    ExcludedData,
}

impl EventType {
    pub fn label(&self) -> &'static str {
        match self {
            EventType::ObstructiveApnea => "Obstructive Apnea",
            EventType::CentralApnea => "Central Apnea",
            EventType::MixedApnea => "Mixed Apnea",
            EventType::UnclassifiedApnea => "Unclassified Apnea",
            EventType::Hypopnea => "Hypopnea",
            EventType::Desaturation => "Desaturation",
            EventType::CheyneStokes => "Cheyne-Stokes",
            EventType::Snore => "Snore",
            EventType::ExcludedData => "Excluded Data",
        }
    }

    /// Is this event counted toward AHI calculation?
    pub fn is_ahi_event(&self) -> bool {
        matches!(
            self,
            EventType::ObstructiveApnea
                | EventType::CentralApnea
                | EventType::MixedApnea
                | EventType::UnclassifiedApnea
                | EventType::Hypopnea
        )
    }
}

/// A detected sleep event with timing and context
#[derive(Debug, Clone, Serialize)]
pub struct SleepEvent {
    pub event_type: EventType,
    pub start_sec: f64,
    pub end_sec: f64,
    pub duration_sec: f64,
    pub position: BodyPosition,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spo2_nadir: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spo2_drop: Option<f64>,
}

impl SleepEvent {
    pub fn new(event_type: EventType, start_sec: f64, end_sec: f64, position: BodyPosition) -> Self {
        Self {
            event_type,
            start_sec,
            end_sec,
            duration_sec: end_sec - start_sec,
            position,
            spo2_nadir: None,
            spo2_drop: None,
        }
    }
}
