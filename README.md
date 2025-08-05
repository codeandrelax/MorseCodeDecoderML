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

  ## Setup

  Run setup_env.sh script for Mac and Linux. (To make it executable use chmod +x setup_env.sh)

  Run setup_env.bat script for Windows.

  ## Example usage

  py .\main.py --input_filename .\cq.wav --plot --smooth_window 1 --interactive_threshold

  Use interactive threshold to manualy set threshold value for signal binarization.

  --smooth_signal paramater applies a moving average filter over the signal with a given window size (smooth_window), therefore reducing the fast changes in the signal. In esence this is a low-pass filter which can be fine tuned depending on a signal you're dealing with. If the signal is quite clear and good keep the value low. Deafult value is 3 which is good for both slightly noisy signals and for good crisp signals.
  