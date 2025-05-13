"""
GenerateMorseAudioExamples.py

This script generates Morse code audio files from random sentences. The sentences are converted into Morse code and then modulated into audio. 
The generated audio files are saved in the 'audio_examples' directory with randomized signal-to-noise ratios (SNR) ranging from 5 to 15.

Author: Damjan Prerad
Date: 2025-05-13
"""

import os
import random
import argparse
from scipy.io import wavfile
import numpy as np
import re
from MorseAlphabet import MORSE_CODE_DICT, ALPHABET
from Text2MorseWav import generate_morse_sample

sentences = [
        "HELLO THERE",
        "HOW ARE YOU",
        "MORSE CODE IS FUN",
        "PYTHON IS AWESOME",
        "I LOVE CODING",
        "HELLO WORLD",
        "LEARNING MACHINE LEARNING",
        "MORSE CODE DECODER",
        "THIS IS A TEST",
        "I AM A PROGRAMMER",
        "GOOD MORNING",
        "WELCOME TO THE FUTURE",
        "LET'S MAKE MUSIC",
        "AI AND MACHINE LEARNING",
        "TECHNOLOGY IS THE FUTURE"
    ]

def main():
    parser = argparse.ArgumentParser(description="Generate random Morse code audio .wav files.")
    parser.add_argument('--sample_rate', type=int, default=2000, help='Sample rate in Hz')
    parser.add_argument('--output_dir', type=str, default='audio_examples', help='Directory to save the generated WAV files')
    parser.add_argument('--wpm', type=int, default=18, help='Words per minute (speed of Morse code)')
    parser.add_argument('--pitch', type=float, default=950, help='Tone frequency in Hz')
    parser.add_argument('--amplitude', type=int, default=100, help='Signal amplitude (0–100)')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate 15 sentences with randomized SNR between 5 and 15
    for i in range(15):
        snr = random.uniform(5, 15)  # Randomize SNR between 5 and 15
        
        modified_sentence = re.sub(r'[^a-zA-Z ]', '', sentences[i])
        print(f"modified_sentence {modified_sentence}")

        time_signal, morse_string = generate_morse_sample(
            sr=args.sample_rate,
            text_len=len(modified_sentence),
            pitch=args.pitch,
            wpm=args.wpm,
            snrDB=snr,
            amplitude=args.amplitude,
            s=modified_sentence,
            apply_random_length_dots=True
        )

        modified_sentence = modified_sentence.replace(' ', '_') 
        output_filename = os.path.join(args.output_dir, f"{modified_sentence}.wav")
        # Convert to 16-bit WAV format and save
        int_signal = (time_signal * 32767).astype(np.int16)
        wavfile.write(output_filename, args.sample_rate, int_signal)

        print(f"Generated '{output_filename}' from sentence: '{sentences[i]}' with SNR: {snr:.2f} dB")

if __name__ == "__main__":
    main()

