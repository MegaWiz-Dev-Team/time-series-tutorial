import numpy as np
import pandas as pd

def extract_epoch_features(epoch_df):
    """
    Extract features from a 30-second epoch DataFrame.
    Assumes standard columns from Level 1: 'Resp_nasal', 'SaO2', 'Pulse'
    """
    features = {}
    
    # Identify actual column names
    flow_col = [c for c in epoch_df.columns if 'nasal' in c.lower()][0]
    sao2_col = [c for c in epoch_df.columns if 'sao2' in c.lower() or 'spo2' in c.lower()][0]
    pulse_col = [c for c in epoch_df.columns if 'pulse' in c.lower()][0]
    
    # Time-domain stats: Flow
    features['flow_mean'] = epoch_df[flow_col].mean()
    features['flow_std'] = epoch_df[flow_col].std()
    features['flow_min'] = epoch_df[flow_col].min()
    features['flow_max'] = epoch_df[flow_col].max()
    features['flow_variance'] = features['flow_std'] ** 2
    
    # Frequency-domain proxy: Zero crossing rate of Flow (mean-centered)
    flow_centered = epoch_df[flow_col] - features['flow_mean']
    zero_crossings = np.where(np.diff(np.sign(flow_centered)))[0]
    features['flow_zcr'] = len(zero_crossings)
    
    # Time-domain stats: SpO2
    features['sao2_mean'] = epoch_df[sao2_col].mean()
    features['sao2_min'] = epoch_df[sao2_col].min()
    features['sao2_drop'] = epoch_df[sao2_col].max() - epoch_df[sao2_col].min()
    
    # Time-domain stats: Pulse
    features['pulse_mean'] = epoch_df[pulse_col].mean()
    features['pulse_std'] = epoch_df[pulse_col].std()
    
    # Cross-channel feature
    corr = epoch_df[flow_col].corr(epoch_df[sao2_col])
    features['flow_sao2_corr'] = corr if not pd.isna(corr) else 0.0
    
    return features
