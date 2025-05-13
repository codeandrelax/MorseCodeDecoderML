
"""
Text2MorseWav.py

This script generates a Morse code audio (.wav) file from a given text input.
It allows the customization of sample rate, signal-to-noise ratio (SNR), words per minute (WPM),
tone frequency, and signal amplitude. It can also add noise to the generated signal for testing purposes.

Author: Damjan Prerad
Date: 2025-05-13

Example usage:
python Text2MorseWav.py --text "HELLO" --sample_rate 2000 --snr 5 --output hello.wav
"""

import numpy as np
import random
from scipy.io import wavfile
import argparse
from MorseAlphabet import MORSE_CODE_DICT, ALPHABET, num_tags, idx_to_tag, latin_to_morse, morse_to_cyrillic_text

sr = 2000 # sample rate

def apply_noise_to_signal(signal, SNRdb):
    """
    Applies noise to the input signal based on a specified Signal-to-Noise Ratio (SNR).

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal to which noise will be added.
    SNRdb : float
        Desired signal-to-noise ratio in decibels (dB).

    Returns
    -------
    numpy.ndarray
        Signal with added noise.
    """
    signal_power = np.mean(abs(signal ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)  # calculate noise power based on signal power and SNR
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

def get_dot(ref, apply_random_length_dots):
    """
    Determines the length of a dot based on the reference value and whether random dot lengths are applied.

    Parameters
    ----------
    ref : int
        Reference length for a dot.
    apply_random_length_dots : bool
        If True, the dot length will vary randomly.

    Returns
    -------
    int
        Length of the dot in samples.
    """
    if apply_random_length_dots:
        scale = np.clip(np.random.normal(1, 0.2), 0.5, 2.0)
        return int(ref * scale)
    else:
        return int(ref)

def get_dash(ref, apply_random_length_dots):
    """
    Determines the length of a dash based on the reference value and whether random dash lengths are applied.

    Parameters
    ----------
    ref : int
        Reference length for a dash.
    apply_random_length_dots : bool
        If True, the dash length will vary randomly.

    Returns
    -------
    int
        Length of the dash in samples.
    """
    if apply_random_length_dots:
        scale = np.clip(np.random.normal(1, 0.2), 0.5, 2.0)
        return int(3 * ref * scale)
    else:
        return int(3 * ref)
    
def generate_morse_sample(sr=2000, text_len=10, pitch=950, wpm=18, snrDB=1, amplitude=100, s=None, apply_random_length_dots = True):
    """
    Generates a Morse code sample as a time-domain audio signal and its corresponding Morse string.

    Parameters
    ----------
    sr : int, optional
        The sample rate (in Hz) of the output audio signal (default is 2000).
    text_len : int, optional
        The length of the text to convert to Morse code (default is 10).
    pitch : int, optional
        The pitch (frequency in Hz) of the Morse tone (default is 950).
    wpm : int, optional
        Words per minute (speed) of the Morse code (default is 18).
    snrDB : float, optional
        Signal-to-noise ratio in decibels (default is 1).
    amplitude : int, optional
        Signal amplitude (range 0-100) (default is 100).
    s : str, optional
        The input string to convert to Morse code. If None, a random string is generated (default is None).
    apply_random_length_dots : bool, optional
        If True, dots and dashes will have random lengths (default is True).

    Returns
    -------
    tuple
        A tuple containing the time-domain audio signal and the generated Morse string.
    """
    assert pitch < sr / 2  # Nyquist

    # Common morse spacing [https://en.wikipedia.org/wiki/Morse_code]
    ref = (60 / wpm) / 50 * sr

    # Create random string that doesn't start or end with a space
    if s is None:
        s1 = ''.join(random.choices(ALPHABET, k=text_len - 2))
        s2 = ''.join(random.choices(ALPHABET[1:], k=2))
        s = s2[0] + s1 + s2[1]

    time_domain_signal = []
    time_domain_signal.append(np.zeros(5 * get_dot(ref, apply_random_length_dots)))

    # dat\dit space = 1 dot, char space = 3 dots, word space = 7 dots
    for c in s:
        if c == ' ':
            time_domain_signal.append(np.zeros(7 * get_dot(ref, apply_random_length_dots)))
        else:
            for m in MORSE_CODE_DICT[c]:
                if m == '.':
                    time_domain_signal.append(np.ones(get_dot(ref, apply_random_length_dots)))
                    time_domain_signal.append(np.zeros(get_dot(ref, apply_random_length_dots)))
                elif m == '-':
                    time_domain_signal.append(np.ones(get_dash(ref, apply_random_length_dots)))
                    time_domain_signal.append(np.zeros(get_dot(ref,apply_random_length_dots)))

            time_domain_signal.append(np.zeros(2 * get_dot(ref, apply_random_length_dots)))

    time_domain_signal.append(np.zeros(5 * get_dot(ref, apply_random_length_dots)))
    time_domain_signal_ = time_domain_signal.copy()
    time_domain_signal = np.hstack(time_domain_signal)

    # Modulation
    t = np.arange(len(time_domain_signal)) / sr
    sine = np.sin(2 * np.pi * t * pitch)
    time_domain_signal = sine * time_domain_signal

    # Add noise
    time_domain_signal = apply_noise_to_signal(time_domain_signal, snrDB)
    time_domain_signal *= amplitude / 100
    time_domain_signal = np.clip(time_domain_signal, -1, 1)
    time_domain_signal = time_domain_signal.astype(np.float32)


    return time_domain_signal, s

def main():
    """
    Main function to parse arguments and generate a Morse code audio file from the input text.

    It saves the resulting audio as a .wav file and prints the generated Morse string.
    """
    parser = argparse.ArgumentParser(description="Generate a Morse code audio .wav file from text.")
    parser.add_argument('--text', type=str, help='Text to convert to Morse code')
    parser.add_argument('--sample_rate', type=int, default=2000, help='Sample rate in Hz')
    parser.add_argument('--snr', type=float, default=1.0, help='Signal-to-noise ratio in dB')
    parser.add_argument('--output', type=str, default='output.wav', help='Output wav file name')
    parser.add_argument('--wpm', type=int, default=18, help='Words per minute (speed of Morse code)')
    parser.add_argument('--pitch', type=float, default=950, help='Tone frequency in Hz')
    parser.add_argument('--amplitude', type=int, default=100, help='Signal amplitude (0–100)')

    args = parser.parse_args()

    filtered_text = ''
    if args.text:
        upper_text = args.text.upper()
        filtered_text = ''.join(char for char in upper_text if char in MORSE_CODE_DICT or char == ' ')

    time_signal, morse_string = generate_morse_sample(
        sr=args.sample_rate,
        text_len=len(filtered_text) if args.text else 10,
        pitch=args.pitch,
        wpm=args.wpm,
        snrDB=args.snr,
        amplitude=args.amplitude,
        s=filtered_text,
        apply_random_length_dots=True
    )

    # Convert to 16-bit WAV
    int_signal = (time_signal * 32767).astype(np.int16)
    wavfile.write(args.output, args.sample_rate, int_signal)

    print(f"Generated '{args.output}' from text: '{morse_string}'")

if __name__ == "__main__":
    main()

