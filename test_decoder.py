import os
import subprocess
import random
import argparse
import sys

# Sample texts (uppercase, no punctuation, sometimes numbers)
sample_texts = [
    "HELLO WORLD",
    "CALL ME AT 123456",
    "PYTHON IS FUN",
    "EMERGENCY 911",
    "SEND HELP NOW",
    "CODE RED ALERT",
    "TEST MESSAGE 001",
    "MEET AT LOCATION 42",
    "RADIO CHECK OVER",
    "UNIT 7 REPORT IN"
]

def generate_samples(save_metadata):
    snr_values = [3, 5, 10, 12, 15]
    sample_rate = 8000
    num_samples_per_snr = 10

    base_dir = "samples"
    os.makedirs(base_dir, exist_ok=True)

    for snr in snr_values:
        snr_dir = os.path.join(base_dir, f"SNR_{snr}")
        os.makedirs(snr_dir, exist_ok=True)

        for i in range(num_samples_per_snr):
            text = random.choice(sample_texts)
            output_filename = f"sample_{i+1}.wav"
            output_path = os.path.join(snr_dir, output_filename)

            command = [
                "python3",
                "generate_morse_audio.py",
                "--text", text,
                "--sample_rate", str(sample_rate),
                "--snr", str(snr),
                "--output", output_path
            ]

            print(f"Generating: {output_path} | Text: \"{text}\"")
            subprocess.run(command, check=True)

            if save_metadata:
                metadata_path = os.path.join(snr_dir, f"sample_{i+1}.txt")
                with open(metadata_path, 'w') as f:
                    f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Morse code audio samples.")
    parser.add_argument(
        "--generate_data",
        action="store_true",
        help="If set, generates audio data."
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="If set, saves the text used for each sample in a .txt file."
    )

    args = parser.parse_args()

    if not args.generate_data:
        print("Data generation not requested. Exiting.")
        sys.exit(0)

    generate_samples(args.save_metadata)
