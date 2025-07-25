
import sys
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

def bandpass_filter(data, fs, lowcut=700, highcut=1100, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, data)

def compute_signal_power(data, rate, window_size_ms=10):
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    window_size = int(rate * window_size_ms / 1000)  # samples per window
    num_windows = len(data) // window_size
    power = []

    for i in range(num_windows):
        window = data[i * window_size : (i + 1) * window_size]
        if window.size == 0:
            power.append(0.0)
        else:
            mean_square = np.mean(window ** 2)
            rms = np.sqrt(mean_square) if mean_square >= 0 else 0.0
            power.append(rms)

    times = np.arange(num_windows) * (window_size / rate)
    return times, np.array(power)

def smooth_power(power, window=3):
    return np.convolve(power, np.ones(window)/window, mode='same')

def kmeans_threshold(power, percentage):
    X = power.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())
    return sum(centers) * percentage / 100

def clean_binary(binary, min_duration=5):
    result = []
    count = 1
    for i in range(1, len(binary)):
        if binary[i] == binary[i-1]:
            count += 1
        else:
            if count >= min_duration:
                result.extend([binary[i-1]] * count)
            else:
                result.extend([0] * count)  # treat short blips as silence
            count = 1
    result.extend([binary[-1]] * count)
    return np.array(result)

def detect_word_gaps_and_split(binary_signal, times, zero_threshold=20):
    zero_sequences = []
    start = None

    for i, val in enumerate(binary_signal):
        if val == 0:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= zero_threshold:
                zero_sequences.append((start, i))  # (start_idx, end_idx)
            start = None

    # Catch trailing zero sequence
    if start is not None and (len(binary_signal) - start) >= zero_threshold:
        zero_sequences.append((start, len(binary_signal)))

    # Compute center times
    center_times = [times[(start + end) // 2] for start, end in zero_sequences]

    center_indices = [(start + end) // 2 for start, end in zero_sequences]

    chunks = []
    prev = 0
    for idx in center_indices:
        chunk = binary_signal[prev:idx]
        if len(chunk) > 0:
            chunks.append(chunk)
        prev = idx

    if prev < len(binary_signal):
        chunks.append(binary_signal[prev:])

    return center_times, chunks


def getSEnergy(data, rate, smoothing_window=3):
    times, power = compute_signal_power(data, rate)
    smoothed_power = smooth_power(power, smoothing_window) 
    return power, smoothed_power, times

def getBinarySignal(smoothed_power, times, percentage=67, otsu_threshold=0, zero_threshold=80):
    if otsu_threshold == 1:
        power_norm = (smoothed_power - np.min(smoothed_power)) / (np.max(smoothed_power) - np.min(smoothed_power))
        threshold = threshold_otsu(power_norm)
        percentage = int(threshold * 100 + 0.5)

    print(f"Percentage {percentage}")

    threshold = kmeans_threshold(smoothed_power, percentage)
    binary = np.where(smoothed_power > threshold, 1, 0)
    binary_clean = clean_binary(binary, min_duration=5)

    center_times, chunks = detect_word_gaps_and_split(binary_clean, times, zero_threshold=zero_threshold)

    return binary_clean, chunks, center_times, threshold

def convert2binary(data, rate, smoothing_window=3, percentage=67, otsu_threshold=0, zero_threshold=80):
    filtered_data = data # bandpass_filter(data, rate) # don't use bandpass filter
    times, power = compute_signal_power(filtered_data, rate)
    smoothed_power = smooth_power(power, smoothing_window)

    if otsu_threshold == 1:
        power_norm = (smoothed_power - np.min(smoothed_power)) / (np.max(smoothed_power) - np.min(smoothed_power))
        threshold = threshold_otsu(power_norm)
        percentage = int(threshold * 100 + 0.5)

    print(f"Binary threshold percentage: {percentage}")

    threshold = kmeans_threshold(smoothed_power, percentage)
    binary = np.where(smoothed_power > threshold, 1, 0)
    binary_clean = clean_binary(binary, min_duration=5)

    center_times, chunks = detect_word_gaps_and_split(binary_clean, times, zero_threshold=zero_threshold)

    return binary_clean, chunks, center_times, power, smoothed_power, times, threshold
