import os
import sys
import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from audio_process import convert2binary
from decode_process import convert2text

from audio_process import getSEnergy, getBinarySignal

import re

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from scipy.signal import spectrogram

import textwrap

def load_audio(path):
    rate, data = wav.read(path)
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    return rate, data

def main():
    parser = argparse.ArgumentParser(description="Read a WAV file and process it.")
    decode_group = parser.add_argument_group("Decoding Options")
    parser.add_argument("--input_filename", type=str, help="Path to the WAV file")
    parser.add_argument("--lang", type=str, choices=["en", "ru"], default="en", help="Language for Morse decoding (en or ru)")
    parser.add_argument(
        "--percentage",
        type=lambda x: 0 <= int(x) <= 100 and int(x) or parser.error("Threshold must be between 0 and 100"),
        default=67,
        help="Threshold percentage (0-100)"
    )
    parser.add_argument("--otsu_threshold", type=int, default=0, help="Automatic percentage calculation. Set to 1 to engage")
    parser.add_argument("--plot", action="store_true", help="Plot diagrams")
    parser.add_argument("--interactive_threshold", action="store_true", help="Launch interactive threshold tuning")
    parser.add_argument(
        "--zero_threshold",
        type=int,
        default=80,
        help="Initial zero threshold value (integer)."
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Width of smoothing window (integer)."
    )

    args = parser.parse_args()

    if args.lang == "ru":
        print("Using Russian Morse Code Dictionary")
    else:
        print("Using English Morse Code Dictionary")

    rate, data = load_audio(args.input_filename)
    print(f"Loaded f{args.input_filename} with sample rate of {rate}")

    if args.interactive_threshold:
        interactive_mode(data, rate, args)

    # data = data[0:rate*10]

    binary_signal, chunks, center_times, power, smoothed_power, times, threshold = convert2binary(data,
                                                                                                  rate,
                                                                                                  smoothing_window=args.smooth_window,
                                                                                                  percentage=args.percentage,
                                                                                                  otsu_threshold=args.otsu_threshold,
                                                                                                  zero_threshold=args.zero_threshold)

    extracted_text = ""
    morse_elements = ""

    chunk_wpms = []
    for chunk in chunks:
        try:
            extracted_text_part, morse_element_part, chunk_wpm = convert2text(chunk, rate, args.lang)

            chunk_wpms.append(chunk_wpm)
            extracted_text += extracted_text_part
            morse_elements += morse_element_part
        except Exception as e:
            print("Failed to process chunk")
            rising_edges = np.where(np.diff(chunk.astype(int)) == 1)[0]
            
            chunk_wpms.append(0)
            for i in range(0, len(rising_edges)):
                morse_elements += 'x'

            print(e)

    cleaned_text = re.sub(r'\?+', ' ', extracted_text)
    cleaned_morse = re.sub(r'/+', '/', morse_elements)

    morse_for_graph = cleaned_morse.replace('/', '').replace(' ', '')

    print(f"Extracted text: {cleaned_text}")
    print(f"Morse dit-dash: {cleaned_morse}")

    print("Words per minute typed per block (wpm):", ", ".join(str(wpm) for wpm in chunk_wpms))

    base_name = os.path.splitext(os.path.basename(args.input_filename))[0]

    if args.plot == True:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        f, t_spec, Sxx = spectrogram(data, fs=rate, nperseg=256, noverlap=0)
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        # Spectrogram
        ax0 = axes[0]
        im = ax0.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='magma')
        ax0.set_ylabel('Frequency [Hz]')
        ax0.set_title('Spectrogram of Original Signal')

        # Raw power
        axes[1].plot(times, power, color='darkred')
        axes[1].set_title('Raw Signal Power')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True)

        # Smoothed + threshold
        axes[2].plot(times, smoothed_power, color='navy')
        axes[2].axhline(threshold, color='red', linestyle='--', label='Threshold')
        # axes[2].plot(times, threshold, color='orange', linestyle='--', label='Dynamic Threshold')
        axes[2].set_title('Smoothed Power + Threshold')
        axes[2].set_ylabel('Amplitude')
        axes[2].legend()
        axes[2].grid(True)

        # Step 4: Plot vertical lines on the last subplot
        # for ct in center_times:
        #     axes[3].axvline(x=ct, color='green', linestyle='--', alpha=0.6, label='Detected Word Gap')
        for i, ct in enumerate(center_times):
            axes[3].axvline(x=ct, color='green', linestyle='--', alpha=0.6)
            if i < len(chunk_wpms):
                wpm = chunk_wpms[i]
                axes[3].text(ct, 1.15 * np.max(smoothed_power), f'{wpm:.1f} wpm',
                            fontsize=10, ha='center', va='bottom', rotation=90, color='green')

        # Binary cleaned
        axes[3].plot(times, binary_signal * np.max(smoothed_power), color='black', alpha=0.8)
        axes[3].set_title('Cleaned Binary Morse Signal', pad=20)
        axes[3].set_xlabel('Time [s]')
        axes[3].set_ylabel('Binary')
        axes[3].grid(True)

        rising_edges = (np.diff(binary_signal.astype(int)) == 1).nonzero()[0] + 1
        num_symbols = min(len(rising_edges), len(morse_for_graph))

        for i in range(num_symbols):
            t = times[rising_edges[i]]
            symbol = morse_for_graph[i]
            axes[3].text(t, 1.05 * np.max(smoothed_power), symbol, fontsize=14, ha='center', va='bottom', color='blue')

        # plt.tight_layout()
        # fig.suptitle(cleaned_text, fontsize=16)
        wrapped_title = "\n".join(textwrap.wrap(cleaned_text, width=60))
        fig.suptitle(wrapped_title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(f"{base_name}.png", dpi=300)
        fig.canvas.manager.set_window_title("Decoded morse")
        plt.show()

    with open(f"{base_name}.txt", "w", encoding="utf-8") as f:
        f.write("Decoded Text:\n")
        f.write(cleaned_text + "\n\n")
        f.write("Decoded Morse:\n")
        f.write(cleaned_morse + "\n")

def interactive_mode(data, rate, args):
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(bottom=0.25)

    f, t_spec, Sxx = spectrogram(data, fs=rate, nperseg=256, noverlap=0)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    ax0 = ax[0]
    im = ax0.pcolormesh(t_spec, f, Sxx_dB, shading='gouraud', cmap='magma')
    ax0.set_ylabel('Freq [Hz]')
    ax0.set_title('Spectrogram')

    raw_line, = ax[1].plot([], [], color='darkred')
    ax[1].set_title('Raw Power')
    ax[1].grid(True)

    smooth_line, = ax[2].plot([], [], color='navy')
    thresh_line, = ax[2].plot([], [], color='red', linestyle='--', label='Threshold')

    ax[2].legend()
    ax[2].set_title('Smoothed Power + Threshold')
    ax[2].grid(True)

    binary_line, = ax[3].plot([], [], color='black')
    ax[3].set_title('Cleaned Binary')
    ax[3].set_xlabel('Time [s]')
    ax[3].grid(True)

    pos = ax[3].get_position()
    ax[3].set_position([pos.x0, pos.y0 - 0.06, pos.width, pos.height])

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, 'Threshold %', 0, 100, valinit=args.percentage, valstep=1)

    def on_zt_submit(text):
        try:
            args.zero_threshold = int(text)
            update(int(slider.val)) 
        except ValueError:
            print("Zero Thresh must be an integer")

    ax_box = plt.axes([0.25, 0.08, 0.1, 0.035])
    zt_text_box = TextBox(ax_box, 'Consecutive zero thresh.:', initial=str(args.zero_threshold))
    zt_text_box.on_submit(on_zt_submit)

    def on_smt_submit(text):
        try:
            args.smooth_window = int(text)
            update(int(slider.val)) 
        except ValueError:
            print("Smoothing window must be an integer")

    ax_box = plt.axes([0.65, 0.08, 0.1, 0.035])
    smt_text_box = TextBox(ax_box, 'Smoothing window:', initial=str(args.smooth_window))
    smt_text_box.on_submit(on_smt_submit)

    text_box = fig.text(0.5, 0.01, "", ha="center", fontsize=12)

    # power, smoothed_power, times = getSEnergy(data, rate)

    def update(val):
        pct = int(slider.val)
        args.percentage = pct
        print(f"Consecutive zero threshold value: {args.zero_threshold}")
        print(f"Smoothing window width: {args.smooth_window}")

        binary_signal, chunks, center_times, power, smoothed_power, times, threshold = convert2binary(
            data, rate, smoothing_window=args.smooth_window, percentage=pct, otsu_threshold=args.otsu_threshold, zero_threshold=args.zero_threshold)

        # binary_signal, chunks, center_times, threshold = getBinarySignal(smoothed_power, times, percentage=pct, otsu_threshold=args.otsu_threshold)

        morse_elements = ""
        all_text = ""
        chunk_wpms = []

        for chunk in chunks:
            try:
                text, morse, wpm = convert2text(chunk, rate, args.lang)
                all_text += text
                morse_elements += morse
                chunk_wpms.append(wpm)
            except Exception as e:
                print(f"[Warning] Failed to decode chunk: {e}")
                morse_elements += 'x' * len(np.where(np.diff(chunk.astype(int)) == 1)[0])
                chunk_wpms.append(0)

        morse_for_graph = re.sub(r'/+', '/', morse_elements).replace('/', '').replace(' ', '')
        cleaned_text = re.sub(r'\?+', ' ', all_text).strip()

        [l.remove() for l in ax[3].lines[1:]]
        [t.remove() for t in ax[3].texts]
        [v.remove() for v in ax[2].texts]

        raw_line.set_data(times, power)
        ax[1].relim(); ax[1].autoscale_view()

        smooth_line.set_data(times, smoothed_power)
        thresh_line.set_data(times, np.full_like(times, threshold))
        ax[2].relim(); ax[2].autoscale_view()

        binary_line.set_data(times, binary_signal * np.max(smoothed_power))
        ax[3].relim(); ax[3].autoscale_view()

        for i, ct in enumerate(center_times):
            ax[3].axvline(x=ct, color='green', linestyle='--', alpha=0.6)
            if i < len(chunk_wpms):
                wpm = chunk_wpms[i]
                ax[3].text(ct, 1.15 * np.max(smoothed_power), f'{wpm:.1f} wpm',
                           fontsize=10, ha='center', va='bottom', rotation=90, color='green')

        rising_edges = (np.diff(binary_signal.astype(int)) == 1).nonzero()[0] + 1
        num_symbols = min(len(rising_edges), len(morse_for_graph))
        for i in range(num_symbols):
            t = times[rising_edges[i]]
            symbol = morse_for_graph[i]
            ax[3].text(t, 1.05 * np.max(smoothed_power), symbol, fontsize=14,
                       ha='center', va='bottom', color='blue')

        # fig.suptitle(f"Decoded: {cleaned_text}", fontsize=14)
        wrapped_title = "\n".join(textwrap.wrap(cleaned_text, width=60))
        fig.suptitle(wrapped_title, fontsize=16)
        text_box.set_text(f"Threshold: {pct}%")

        fig.canvas.draw_idle()

    update(args.percentage)
    slider.on_changed(update)
    fig.canvas.manager.set_window_title("Decoded morse")
    
    plt.show()

if __name__ == "__main__":
    main()
