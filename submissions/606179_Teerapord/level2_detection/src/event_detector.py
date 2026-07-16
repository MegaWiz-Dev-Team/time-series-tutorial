import numpy as np
import pandas as pd
from scipy.ndimage import label

class SleepEventDetector:
    def __init__(self, fs=10):
        self.fs = fs
        
    def _get_envelope(self, series, window_s=2):
        """Calculate breathing amplitude envelope using rolling standard deviation."""
        window_samples = int(window_s * self.fs)
        # using std as a proxy for amplitude
        return series.rolling(window=window_samples, center=True).std()
        
    def _get_baseline(self, series, window_s=120, method='mean'):
        """Calculate local baseline."""
        window_samples = int(window_s * self.fs)
        if method == 'mean':
            return series.rolling(window=window_samples, center=True, min_periods=1).mean()
        elif method == 'max':
            return series.rolling(window=window_samples, center=True, min_periods=1).max()
        elif method == 'percentile_90':
            return series.rolling(window=window_samples, center=True, min_periods=1).quantile(0.9)
            
    def get_continuous_segments(self, mask, min_duration_s=10):
        """Find continuous True segments in a boolean mask that last >= min_duration_s."""
        min_samples = int(min_duration_s * self.fs)
        labeled, num_features = label(mask)
        
        events = []
        for i in range(1, num_features + 1):
            idx = np.where(labeled == i)[0]
            if len(idx) >= min_samples:
                events.append({
                    'start_idx': idx[0],
                    'end_idx': idx[-1],
                    'duration_s': len(idx) / self.fs
                })
        return events

    def detect_apneas(self, flow_series, min_duration_s=10, drop_threshold=0.90):
        """Detect apneas: Airflow drop >= 90% for >= 10s."""
        envelope = self._get_envelope(flow_series)
        baseline = self._get_baseline(envelope, window_s=120, method='mean')
        
        # Drop >= 90% means envelope <= 10% of baseline
        apnea_mask = envelope <= (1.0 - drop_threshold) * baseline
        # Fill NAs with False
        apnea_mask = apnea_mask.fillna(False)
        
        events = self.get_continuous_segments(apnea_mask, min_duration_s)
        for e in events:
            e['type'] = 'Apnea'
        return events
        
    def detect_desaturations(self, sao2_series, drop_threshold=3.0):
        """Detect desaturations: SpO2 drop >= 3% from baseline."""
        baseline = self._get_baseline(sao2_series, window_s=120, method='percentile_90')
        
        desat_mask = sao2_series <= (baseline - drop_threshold)
        desat_mask = desat_mask.fillna(False)
        
        # Desaturations don't have a strict minimum duration, but let's say >= 3s to avoid noise
        events = self.get_continuous_segments(desat_mask, min_duration_s=3)
        for e in events:
            e['type'] = 'Desaturation'
            e['lowest_spo2'] = sao2_series.iloc[e['start_idx']:e['end_idx']+1].min()
            e['baseline_spo2'] = baseline.iloc[e['start_idx']]
        return events

    def detect_hypopneas(self, flow_series, sao2_series, min_duration_s=10, drop_threshold=0.30, spo2_drop=3.0):
        """Detect hypopneas: Airflow drop >= 30% for >= 10s + SpO2 drop >= 3%."""
        envelope = self._get_envelope(flow_series)
        baseline = self._get_baseline(envelope, window_s=120, method='mean')
        
        # Flow drop >= 30% means envelope <= 70% of baseline
        # We also want to exclude regions that are Apneas (drop >= 90%)
        hypopnea_flow_mask = (envelope <= (1.0 - drop_threshold) * baseline) & (envelope > 0.10 * baseline)
        hypopnea_flow_mask = hypopnea_flow_mask.fillna(False)
        
        flow_events = self.get_continuous_segments(hypopnea_flow_mask, min_duration_s)
        
        desat_events = self.detect_desaturations(sao2_series, drop_threshold=spo2_drop)
        
        # Associate flow events with desaturations
        # A valid hypopnea must have a desaturation overlapping or starting within 30s of the flow drop
        hypopnea_events = []
        for fe in flow_events:
            associated = False
            for de in desat_events:
                # Check if desaturation starts during or shortly after the flow event
                # Allow desaturation to start up to 30s after flow event starts
                max_delay_samples = 30 * self.fs
                if fe['start_idx'] <= de['end_idx'] and de['start_idx'] <= (fe['end_idx'] + max_delay_samples):
                    associated = True
                    fe['lowest_spo2'] = de['lowest_spo2']
                    break
            
            if associated:
                fe['type'] = 'Hypopnea'
                hypopnea_events.append(fe)
                
        return hypopnea_events

    def calculate_ahi(self, events, total_duration_s, df=None, position_col=None):
        """Calculate AHI and Positional AHI."""
        total_hours = total_duration_s / 3600.0
        
        num_apneas = sum(1 for e in events if e['type'] == 'Apnea')
        num_hypopneas = sum(1 for e in events if e['type'] == 'Hypopnea')
        
        ahi = (num_apneas + num_hypopneas) / total_hours if total_hours > 0 else 0
        
        report = {
            'Total Apneas': num_apneas,
            'Total Hypopneas': num_hypopneas,
            'Total Events': num_apneas + num_hypopneas,
            'Total Hours': total_hours,
            'AHI': ahi
        }
        
        # Positional breakdown
        if df is not None and position_col is not None:
            # Assume Position == 0 is Supine, others are Non-supine
            # First, calculate total time in each position
            # Count valid samples
            valid_mask = df[position_col].notna()
            supine_time_s = (df[position_col] == 0).sum() / self.fs
            nonsupine_time_s = (df[position_col] > 0).sum() / self.fs
            
            supine_events = 0
            nonsupine_events = 0
            
            for e in events:
                if e['type'] in ['Apnea', 'Hypopnea']:
                    # Get the most frequent position during the event
                    pos_segment = df[position_col].iloc[e['start_idx']:e['end_idx']]
                    if len(pos_segment) > 0:
                        dom_pos = pos_segment.mode()[0]
                        if dom_pos == 0:
                            supine_events += 1
                        else:
                            nonsupine_events += 1
                            
            report['Supine Hours'] = supine_time_s / 3600.0
            report['Non-supine Hours'] = nonsupine_time_s / 3600.0
            
            report['Supine AHI'] = supine_events / report['Supine Hours'] if report['Supine Hours'] > 0 else 0
            report['Non-supine AHI'] = nonsupine_events / report['Non-supine Hours'] if report['Non-supine Hours'] > 0 else 0
            
        return report
