# MorseCodeDecoderML
Morse Code Decoder using machine learning.
Machine learning elements are done using pytorch.

# Requirements
- Input data: raw wav file
- Output data:
  1. text file with decoded information
  2. text file with dashes and dots / decoded information
  3. Image of processed spectrogram

# Solution approach
Two models are trained for this purpose. One for the purpooses of morse code decoding (LSTM network) and second for purposes of morse code detection / framing (resnet).


# Dataset

Morse code audio file is produced after which it's spectrogram is generated.
This spectrogram is then used as an input to models.

![image](https://github.com/user-attachments/assets/22fd50f7-2735-47f0-8570-9f8a9540b0bf)
