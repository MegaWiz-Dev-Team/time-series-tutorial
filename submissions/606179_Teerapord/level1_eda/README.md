# Level 1: Time Series EDA & Signal Quality

## Overview
This submission contains the Exploratory Data Analysis (EDA) and data cleaning pipeline for Home Sleep Test (HST) signals. The goal is to evaluate the signal quality and output a clean, uniform dataset ready for downstream Event Detection (Level 2).

## File Structure
- `Level1_EDA.ipynb`: The main notebook containing the entire EDA workflow.
- `requirements.txt`: Python dependencies required to run the code.
- `outputs/clean_data.parquet`: The processed and aligned dataset (10 Hz) with sentinels removed.
- `outputs/events_timeline.png`: A 5-minute snapshot showing multi-channel alignment.

## Approach
1. **Data Loading**: Extracted the EDF from the raw patient directory.
2. **Resampling**: To create a unified DataFrame, all signals (100 Hz, 10 Hz, 1 Hz) were resampled to **10 Hz** using linear interpolation.
3. **Data Cleaning**: Identified sentinel values (Pulse=511, SaO2=127) and replaced them with `NaN`.
4. **Signal Quality Index (SQI)**: Calculated the percentage of valid data for each channel.
5. **Correlation**: Showed that reduced flow variance often precedes a drop in SaO2 by roughly 20 seconds.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook: `jupyter notebook Level1_EDA.ipynb`
3. Run all cells to process the data and generate plots in the `outputs/` folder.

## Key Findings
- **Data Uniformity**: Resampling all channels to a uniform **10 Hz** frequency effectively synchronizes the multi-modal sensors, resulting in a clean, structured dataset (235,900 samples) ready for tabular or time-series modeling.
- **Sentinel Values Identified**: The `Pulse` channel recorded 9,986 sentinel values (511) and the `SaO2` channel recorded 9,806 sentinel values (127). These anomalous readings were successfully detected and replaced with `NaN`.
- **High Signal Quality**: The overall Signal Quality Index (SQI) for essential channels is excellent at **95.70%**. Respiration, position, and accelerometer channels have 100% valid data, while SpO2 and Pulse sensors maintain a high ~95.8% SQI despite temporary connection losses.
