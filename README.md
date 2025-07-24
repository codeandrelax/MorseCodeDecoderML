# Morse Code Audio Transcription with ML (K-means clustering, otsu threshold and GaussianMixture)

<img width="1185" height="946" alt="image" src="https://github.com/user-attachments/assets/d57affea-37a3-4a02-8530-3ee60fb3a0e2" />


## Usage
usage: main.py [-h] [--input_filename INPUT_FILENAME] [--lang {en,ru}] [--percentage PERCENTAGE] [--otsu_threshold OTSU_THRESHOLD] [--plot] [--interactive_threshold]

Read a WAV file and process it.

options:
  -h, --help            show this help message and exit
  --input_filename INPUT_FILENAME
                        Path to the WAV file
  --lang {en,ru}        Language for Morse decoding (en or ru)
  --percentage PERCENTAGE
                        Threshold percentage (0-100)
  --otsu_threshold OTSU_THRESHOLD
                        Automatic percentage calculation. Set to 1 to engage
  --plot                Plot diagrams
  --interactive_threshold
                        Launch interactive threshold tuning
