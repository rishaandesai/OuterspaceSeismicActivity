import numpy as np
import pandas as pd
from obspy import read
from scipy.signal import butter, filtfilt
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib

# Function to eliminate power frequency noise
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

# Define CNN-LSTM model in PyTorch
class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=71998, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        return x

# Train a CNN-LSTM model (for demonstration purposes)
def train_cnn_lstm_model(training_files):
    sequences = []
    labels = []
    
    for filepath, label in training_files:
        st = read(filepath)
        tr = st[0]
        tr_data = eliminate_power_frequency(tr.data, tr.stats.sampling_rate)
        sequences.append(tr_data)
        labels.append(label)
    
    # Pad sequences to the same length
    max_length = max([len(seq) for seq in sequences])
    sequences_padded = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in sequences])
    labels = np.array(labels)

    # Normalize the data
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences_padded)

    # Reshape data for CNN-LSTM input
    sequences_scaled = sequences_scaled[:, np.newaxis, :]

    # Convert data to PyTorch tensors
    X = torch.tensor(sequences_scaled, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Define model, loss function, and optimizer
    model = CNNLSTMModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model and scaler
    torch.save(model.state_dict(), 'seismic_event_cnn_lstm.pth', _use_new_zipfile_serialization=False)
    joblib.dump(scaler, 'scaler.pkl')

# Train the model before running the analysis
training_files = [
    ('data/mars/training/data/XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed', 1),
    # Add more training files and labels here as needed
]
train_cnn_lstm_model(training_files)

# Main analysis function
def analysis(filepath):
    # Load trained model and scaler
    model = CNNLSTMModel()
    try:
        model.load_state_dict(torch.load('seismic_event_cnn_lstm.pth', map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        print("Model or scaler not found. Please train the model first using train_cnn_lstm_model().")
        return

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

    # Prepare data for model prediction
    tr_data_filt_cleaned = scaler.transform(tr_data_filt_cleaned.reshape(1, -1))
    tr_data_filt_cleaned = torch.tensor(tr_data_filt_cleaned[:, np.newaxis, :], dtype=torch.float32)

    # Predict using the trained model
    with torch.no_grad():
        prediction = model(tr_data_filt_cleaned).item()
    
    if prediction < 0.5:
        print("No seismic event detected.")
        return
    else:
        print("Seismic event detected.")

    # Correct the trigger line placement
    max_vel_time = tr_times_filt[np.argmax(tr_data_filt_cleaned.numpy().flatten())]
    best_trigger_time_abs = tr_filt.stats.starttime + timedelta(seconds=max_vel_time)
    best_trigger_time_str = best_trigger_time_abs.strftime('%Y-%m-%dT%H:%M:%S.%f')

    print(f"Trigger Time: {best_trigger_time_str} at relative time {max_vel_time}")

    # Plot the graph for the **filtered** data with the Best Trigger On line
    plt.figure(figsize=(12, 6))
    plt.plot(tr_times_filt, tr_data_filt_cleaned.numpy().flatten(), label="Filtered Data")
    plt.axvline(x=max_vel_time-40, color='green', label='Best Trigger On')
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
    plt.axvline(x=max_vel_time-40, color='green', label='Best Trigger On')
    plt.xlim([min(tr_times_original), max(tr_times_original)])
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.title(f'{filepath} - Original Data', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
filepath = 'data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-01-19HR00_evid00002.mseed'
analysis(filepath)