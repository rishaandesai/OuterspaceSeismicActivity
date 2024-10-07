import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import find_peaks, butter, filtfilt
from datetime import timedelta
import matplotlib.pyplot as plt

def eliminate_power_frequency(signal, fs, power_freqs=[50, 60], bandwidth=2):
    nyquist = 0.5 * fs
    cleaned_signal = signal.copy()

    for power_freq in power_freqs:
        if power_freq < nyquist:
            low = max(0.1, (power_freq - bandwidth / 2)) / nyquist
            high = min(0.99, (power_freq + bandwidth / 2) / nyquist)
            b, a = butter(2, [low, high], btype='bandstop')
            cleaned_signal = filtfilt(b, a, cleaned_signal)
    return cleaned_signal

def analysis(filepath):
    # Read in seismic data using ObsPy
    st = read(filepath)

    # Extract the original data
    tr_original = st[0]  # Extract the trace
    tr_times_original = tr_original.times()
    tr_data_original = tr_original.data

    # Apply bandpass filter for cleaned data
    minfreq = 0.2
    maxfreq = 2.0
    tr_filt = st.copy().filter('bandpass', freqmin=minfreq, freqmax=maxfreq)[0]
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    # Eliminate power frequencies (50Hz, 60Hz)
    sampling_rate = tr_filt.stats.sampling_rate
    tr_data_filt_cleaned = eliminate_power_frequency(tr_data_filt, sampling_rate)

    # STA/LTA window lengths
    sta_len, lta_len = 60, 600
    print(f"Running STA={sta_len}, LTA={lta_len}")

    # Implement STA/LTA detection algorithm
    df = tr_filt.stats.sampling_rate
    cft = classic_sta_lta(tr_data_filt_cleaned, int(sta_len * df), int(lta_len * df))

    # Find peaks in the STA/LTA characteristic function
    cft_peaks, _ = find_peaks(cft, height=2.0)  # Adjust the threshold for peak detection
    if len(cft_peaks) == 0:
        print(f"No peaks found for STA={sta_len}, LTA={lta_len}.")
        return

    cft_peak_times = tr_times_filt[cft_peaks]
    cft_peak_values = cft[cft_peaks]

    # Find the peak corresponding to the maximum amplitude in the velocity data
    max_vel_time = tr_times_filt[np.argmax(tr_data_filt_cleaned)]
    valid_peaks_indices = np.where(np.abs(cft_peak_times - max_vel_time) <= 500)[0]  # Window size around peak

    if len(valid_peaks_indices) == 0:
        print(f"No valid STA/LTA peak found for STA={sta_len}, LTA={lta_len}.")
        return

    # Find the highest STA/LTA peak within the window
    highest_peak_index_in_window = valid_peaks_indices[np.argmax(cft_peak_values[valid_peaks_indices])]
    closest_trigger_time = cft_peak_times[highest_peak_index_in_window]

    # Correct the trigger line placement
    best_trigger_time_abs = tr_filt.stats.starttime + timedelta(seconds=closest_trigger_time)
    best_trigger_time_str = best_trigger_time_abs.strftime('%Y-%m-%dT%H:%M:%S.%f')

    print(f"Trigger Time: {best_trigger_time_str} at relative time {closest_trigger_time}")

    # Plot the graph for the **filtered** data with the Best Trigger On line
    plt.figure(figsize=(12, 6))
    plt.plot(tr_times_filt, tr_data_filt_cleaned, label="Filtered Data")
    plt.axvline(x=closest_trigger_time, color='green', label='Best Trigger On')
    plt.xlim([min(tr_times_filt), max(tr_times_filt)])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title(f'{filepath} - Filtered Data', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the graph for the **original** data with the Best Trigger On line
    plt.figure(figsize=(12, 6))
    plt.plot(tr_times_original, tr_data_original, label="Original Data")
    plt.axvline(x=closest_trigger_time, color='green', label='Best Trigger On')
    plt.xlim([min(tr_times_original), max(tr_times_original)])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title(f'{filepath} - Original Data', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
filepath = 'data/mars/training/data/XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed'
analysis(filepath)