# Morse Code Audio Transcription with Deep Learning

This project provides a full pipeline to transcribe Morse code audio signals into text, while also generating spectrogram images overlaid with detected Morse regions. Two deep learning models are used in tandem: one to decode the Morse sequence and another to detect where Morse code symbols appear in the spectrogram.

## Project Overview
- Convert a text sentence into Morse audio (Text2MorseWav.py).
- Decode a Morse audio file into text (MorseAudio2Text.py).
- Visualize the Morse detections on the spectrogram and save them as PNGs.
- Save the decoded transcription into a .txt file.
- Use Latin or Cyrillic output script for transcribed text.
- Uses two trained models:
  -   A BiLSTM-based decoder with 2.5% character error rate (CER).
  -   A Faster R-CNN detector to locate Morse symbols in spectrograms.

## Model Architectures
### BiLSTM Decoder
This model is based on a Bidirectional Long Short-Term Memory (BiLSTM) architecture. BiLSTMs are capable of learning long-range dependencies in sequential data by processing the input in both forward and backward directions, making them particularly well-suited for audio-based sequence modeling. For more on BiLSTMs, see G[raves et al., 2005](https://www.cs.toronto.edu/~graves/phd.pdf).

    Achieves 2.5% Character Error Rate (CER) on test data.

    Trained on spectrograms with varying SNR levels and fixed WPM (18).

### Faster R-CNN Detector
The detection model is based on the Faster R-CNN architecture using a ResNet backbone. It is widely used for object detection tasks due to its accuracy and speed. The model detects the positions of Morse symbols (dots, dashes) within spectrogram images.

For more information: [Ren et al., 2015 - Faster R-CNN](https://arxiv.org/abs/1506.01497)

## Directory Structure
```
├── audio_examples/              # Contains generated Morse audio samples (.wav)
├── output_files/                # Output spectrograms and text files
├── MorseAudio2Text.py           # Main script: audio -> text + optional visualization
├── Text2MorseWav.py             # Converts text into Morse code WAV audio
├── GenerateMorseAudioExamples.py# Generates a batch of Morse audio files
├── MorseDecoderArchitecture.py  # Contains BiLSTM model definition
├── MorseAlphabet.py             # Morse code mappings and utilities
├── morse_decode_model.pt        # Trained BiLSTM model weights
├── morse_detect_model.pth       # Trained Faster R-CNN detection model weights
└── __pycache__/                 # Cached Python bytecode
```

## MorseAudio2Text: Audio to Transcription

Transform Morse audio files into transcribed text and visualizations.

```
Arguments
Option	Description
--input_file	Path to the .wav file
--play	1 to play audio, 0 to skip
--decoder_model_path	Path to BiLSTM model (.pt)
--detection_model_path	Path to detection model (.pth)
--show_spec	Display spectrogram plot
--save_spec_image	Save the spectrogram as .png with overlay
--save_txt	Save the transcription as .txt
--output_dir	Directory to save output files
--output_script	Choose between latin or cyrillic output (default: latin)
```

### Example usage
```
python3 MorseAudio2Text.py \
  --input_file audio_examples/WELCOME_TO_THE_FUTURE.wav \
  --play 1 \
  --decoder_model_path morse_decode_model.pt \
  --detection_model_path morse_detect_model.pth \
  --output_script latin \
  --save_txt \
  --save_spec_image \
  --output_dir output_files
```

## Generate Morse Code Audio from Text
Convert sentences into Morse code audio using `Text2MorseWav.py`.

**Example:**
```
python Text2MorseWav.py --text "HELLO THERE" --sample_rate 2000 --snr 10 --output audio_examples/hello_there.wav
```
Or use the generation script:
```
python GenerateMorseAudioExamples.py
```
This will generate 15 audio examples from predefined sentences and save them under `audio_examples/`.

## Training details
Both models were trained on spectrograms derived from audio with varying SNRs.

The words per minute (WPM) rate is fixed at 18 WPM for all training and test data.

Future versions may allow dynamic WPM handling.

## Requirements
- Python 3.8+
- PyTorch
- torchaudio
- matplotlib
- scipy
- numpy
- torchvision
- OpenCV
- pygame (for audio playback)

Install dependencies:
```
pip install -r requirements.txt
```
# Model Files

The trained model files are not included in this repository due to their size. You will need to download them manually before running the transcription scripts.

Please download the following files and place them in the root directory of the project:

    morse_decode_model.pt — the BiLSTM model for Morse code transcription

    morse_detect_model.pth — the Faster R-CNN model for Morse code detection on spectrograms

Download link: 

Once downloaded, ensure the paths provided to the --decoder_model_path and --detection_model_path arguments in the scripts correctly point to these files.

# Dataset

Morse code audio file is produced after which it's spectrogram is generated.
This spectrogram is then used as an input to models.

![image](https://github.com/user-attachments/assets/22fd50f7-2735-47f0-8570-9f8a9540b0bf)
