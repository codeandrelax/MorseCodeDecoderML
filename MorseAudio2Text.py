"""
MorseAudio2Text.py

This script processes a Morse code audio file (.wav), optionally plays it,
generates a spectrogram, uses a deep learning model to decode it into text,
and optionally visualizes Morse detections using an object detection model.

Author: Damjan Prerad
Date: 2025-05-13

Example usage:
python morse_decoder.py --input_file example.wav \
                        --decoder_model_path decoder.pth \
                        --detection_model_path detector.pth \
                        --save_txt --save_spec_image
"""

import argparse
import pygame
import torch
from scipy import signal
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont

from MorseDecoderArchitecture import BiLSTMRnn
from MorseAlphabet import MORSE_CODE_DICT, ALPHABET, num_tags, idx_to_tag, latin_to_morse, morse_to_cyrillic_text

def play_wav(filename):
    """
    Plays a WAV audio file using pygame in the background.

    Parameters
    ----------
    filename : str
        Path to the WAV file to be played.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        print(f"Playing '{filename}' in background...")
    except Exception as e:
        print(f"Error playing '{filename}': {e}")

def load_decoder_model(model_path, device):
    """
    Loads the BiLSTM decoder model from the specified path.

    This function initializes a BiLSTM-based decoder model with a fixed input spectrogram size,
    loads the model weights from disk, and prepares it for inference.

    Parameters
    ----------
    model_path : str
        Path to the trained decoder model file (typically a .pt file).
    device : torch.device
        The device (CPU or CUDA) to map the model to.

    Returns
    -------
    BiLSTMRnn
        The loaded and initialized decoder model in evaluation mode.
    """
    print(f"Loading decoder model from: {model_path}")

    dummy_spectrogram_size = 21
    decoder_model = BiLSTMRnn(num_tags, dummy_spectrogram_size).to(device)
    decoder_model.load_state_dict(torch.load(model_path, map_location=device))
    decoder_model.eval()
    return decoder_model

def load_detector_model(model_path, device):
    """
    Loads the Faster R-CNN detector model from the specified path.

    This function initializes a Faster R-CNN model with a ResNet-50 FPN backbone,
    sets the number of output classes to 2 (e.g. background and Morse object),
    loads the trained weights from the specified checkpoint, and prepares the model for inference.

    Parameters
    ----------
    model_path : str
        Path to the trained detector model file (typically a .pth file).
    device : torch.device
        The device (CPU or CUDA) to map the model to.

    Returns
    -------
    torch.nn.Module
        The loaded and initialized detector model in evaluation mode.
    """
    print(f"Loading detector model from: {model_path}")

    detector_model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    detector_model.load_state_dict(checkpoint)
    detector_model.to(device)
    detector_model.eval()
    return detector_model

def get_spectrogram(samples, sr):
    """
    Computes the spectrogram of the input audio signal.

    This function generates a spectrogram using a short-time Fourier transform (STFT)
    with a window size of 20 milliseconds and no overlap between windows.

    Parameters
    ----------
    samples : np.ndarray
        The raw audio signal as a 1D NumPy array.
    sr : int
        The sample rate of the audio signal (in Hz).

    Returns
    -------
    np.ndarray
        The spectrogram (magnitude) as a 2D NumPy array with shape (freq_bins, time_bins).
    """
    _, _, sxx = signal.spectrogram(samples, nperseg=int(0.02 * sr) , noverlap=0)
    return sxx

def plot_spectrogram(spectrogram, sr):
    """
    Plots a spectrogram using a logarithmic power scale (dB).

    Parameters
    ----------
    spectrogram : np.ndarray
        A 2D NumPy array representing the magnitude spectrogram (frequency bins × time bins).
    sr : int
        The sample rate of the original audio signal (in Hz). Currently unused but included for future extensibility.

    Notes
    -----
    - The power is converted to decibels using `10 * log10`.
    - A small constant (1e-10) is added to avoid log of zero.
    - The plot uses a 'magma' colormap with time on the x-axis and frequency on the y-axis.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(10 * np.log10(spectrogram + 1e-10), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time bins')
    plt.ylabel('Frequency bins')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()

def spectrogram_to_image(spec, upscale_factor=32):
    """
    Converts a spectrogram to a color image using logarithmic scaling and colormap application.

    Parameters
    ----------
    spec : np.ndarray
        The input spectrogram as a 2D NumPy array (frequency bins × time bins).
    upscale_factor : int, optional
        Factor by which to upscale the spectrogram in both dimensions (default is 32).

    Returns
    -------
    spec_color : np.ndarray
        A color image (in BGR format) generated from the spectrogram, as a NumPy array of dtype uint8.

    Notes
    -----
    - The spectrogram is upscaled using `np.repeat` to make it visually larger.
    - A logarithmic transformation (`log1p`) is applied to enhance dynamic range visibility.
    - The result is normalized to the range [0, 255] and converted to an 8-bit image.
    - OpenCV's 'PLASMA' colormap is applied for better visual contrast.
    """
    spec_upscaled = np.repeat(spec, upscale_factor, axis=0)
    spec_upscaled = np.repeat(spec_upscaled, upscale_factor, axis=1)

    spec_log = np.log1p(spec_upscaled)

    spec_norm = (spec_log - np.min(spec_log)) / (np.max(spec_log) - np.min(spec_log))
    spec_img = (spec_norm * 255).astype(np.uint8)

    spec_color = cv2.applyColorMap(spec_img, cv2.COLORMAP_PLASMA)

    return spec_color

def inference_to_str(seq):
    """
    Converts a sequence of predicted indices into a cleaned string output.

    Parameters
    ----------
    seq : list or np.ndarray
        The sequence of predicted indices. If it's a NumPy array, it will be converted to a list.

    Returns
    -------
    str
        A decoded and cleaned string representing the predicted output.

    Notes
    -----
    - Consecutive duplicate indices are collapsed (e.g., [1,1,2,2,2,3] → [1,2,3]).
    - Zeros (assumed to be padding or blank tokens) are removed.
    - Remaining indices are mapped to string characters using `idx_to_tag` and concatenated.
    - Trailing whitespace is stripped from the final result.

    Requires
    --------
    idx_to_tag : dict
        A global dictionary mapping index values to characters or string tokens.
    """
    if not isinstance(seq, list):
        seq = seq.tolist()

    seq = [i[0] for i in groupby(seq)]
    seq = [s for s in seq if s != 0]
    seq = "".join(idx_to_tag[c] for c in seq)
    seq = seq.rstrip()

    return seq

def draw_wrapped_text(img, text, max_width, font_size=32, margin=10, line_spacing=1.5):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Try to load a font with Cyrillic support
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux
        "C:/Windows/Fonts/arial.ttf",                           # Windows
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, font_size)
            break

    if font is None:
        raise RuntimeError("No compatible font found. Please install DejaVuSans or Arial.")

    # Word wrapping
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        if line_width <= max_width - 2 * margin:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Draw lines on image
    x = margin
    y = margin
    for line in lines:
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        # White background rectangle
        draw.rectangle([x - 5, y - 5, x + line_width + 5, y + line_height + 5], fill=(255, 255, 255))
        # Black text
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += int(line_height * line_spacing)

    # Convert back to OpenCV image (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Transform morse audio file into text.")
    parser.add_argument("--input_file", required=True, help="Path to the WAV file to play")
    parser.add_argument("--play", type=int, choices=[0, 1], default=1,
                        help="Set to 1 to play the sound (default), 0 to skip playback")
    parser.add_argument("--decoder_model_path", type=str, required=True, help="Path to decoder ML model")
    parser.add_argument("--detection_model_path", type=str, required=True, help="Path to detection ML model")
    parser.add_argument("--show_spec", action="store_true",
                        help="Set this flag to show the spectrogram plot")
    parser.add_argument("--save_spec_image", action="store_true",
                        help="Set this flag to save the spectrogram image with detected morse code as PNG")
    parser.add_argument("--save_txt", action="store_true",
                        help="Set this flag to save the inferred string to a text file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--output_script", type=str, choices=["latin", "cyrillic"], default="latin",
                        help="Specify the script for output text. Choose 'latin' or 'cyrillic' (default: latin)")
    args = parser.parse_args()

    try:
        import torch, numpy, matplotlib, cv2, scipy
    except ImportError as e:
        raise ImportError(f"Missing package: {e.name}. Please install required dependencies.")

    assert os.path.isfile(args.input_file), f"File '{args.input_file}' does not exist."
    assert args.input_file.lower().endswith(".wav"), "Input file must be a .wav file."
    assert os.path.isfile(args.decoder_model_path), f"Decoder model file '{args.decoder_model_path}' does not exist."
    assert os.path.isfile(args.detection_model_path), f"Detection model file '{args.detection_model_path}' does not exist."

    sr, samples = wavfile.read(args.input_file)
    assert samples.ndim in (1, 2), f"Expected 1D or 2D audio data, but got shape: {samples.shape}"
    if samples.ndim == 2:
        print("Stereo audio detected. Taking mean of both channels.")
        samples = samples.mean(axis=1)
        
    print(f"Sample rate of the loaded file is {sr}. Make sure the model you're doing inference with is trained on the data with the same sample rate as the input file.")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.play == 1:
        play_wav(args.input_file)
    else:
        print(f"Playback skipped for '{args.input_file}'.")

    try:
        decoder_model = load_decoder_model(args.decoder_model_path, device)
    except Exception as e:
        print(f"Error loading decoder model: {e}")
        return
    
    full_spec = get_spectrogram(samples, sr)

    if args.show_spec:
        plot_spectrogram(full_spec, sr)

    # Decode step
    spec_lstm = torch.from_numpy(full_spec).permute(1, 0).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            y_pred_lstm = decoder_model(spec_lstm)
        infered_str = inference_to_str(torch.argmax(y_pred_lstm[0], 1))
    except Exception as e:
        print(f"Inference error: {e}")
        return

    if args.output_script == "cyrillic":
        morse_code = latin_to_morse(infered_str)
        infered_str = morse_to_cyrillic_text(morse_code)

    print(f"Infered sting is {infered_str}")

    if args.save_txt:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        print(f"Base name is {base_name}")
        output_file = os.path.join(args.output_dir, f"{base_name}.txt")

        with open(output_file, 'w') as file:
            file.write(f"Inferred string: {infered_str}\n")

    # Detection step
    try:
        detection_model = load_detector_model(args.detection_model_path, device)
    except Exception as e:
        print(f"Error loading detection model: {e}")
        return

    if args.save_spec_image:
        CHUNK_SIZE = 5000
        OVERLAP = CHUNK_SIZE // 2

        processed_images = []

        for i in range(0, len(samples), OVERLAP):
            chunk = samples[i:i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                break

            chunk_spec = get_spectrogram(chunk, sr)
            chunk_spec_img = spectrogram_to_image(chunk_spec)
            chunk_spec_tensor = F.to_tensor(chunk_spec_img)

            with torch.no_grad():
                detection_bbx = detection_model([chunk_spec_tensor.to(device)])[0]

            boxes = detection_bbx['boxes'].cpu()
            scores = detection_bbx['scores'].cpu()

            if len(scores) > 0:
                max_idx = torch.argmax(scores)
                max_box = boxes[max_idx]
                x1, y1, x2, y2 = map(int, max_box.tolist())

                cv2.rectangle(chunk_spec_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(chunk_spec_img, f"{scores[max_idx]:.2f}", (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # cv2.imshow("Chunk Detection", chunk_spec_img)
                # cv2.waitKey(300) 
            processed_images.append(chunk_spec_img)

        final_image = np.concatenate(processed_images, axis=1)

        MAX_WIDTH = 1920
        MAX_HEIGHT = 1080

        height, width = final_image.shape[:2]
        scaling_factor = min(MAX_WIDTH / width, MAX_HEIGHT / height, 1.0)  # don't upscale

        orig_height, orig_width = final_image.shape[:2]

        if scaling_factor < 1.0:
            new_width = int(width * scaling_factor)
            new_height = height
            final_image = cv2.resize(final_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        height, width = final_image.shape[:2]

        final_image = draw_wrapped_text(final_image, infered_str, final_image.shape[1])
    
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.png")
        cv2.imwrite(output_path, final_image)
        print(f"Saved as {infered_str}.png")

if __name__ == "__main__":
    main()
