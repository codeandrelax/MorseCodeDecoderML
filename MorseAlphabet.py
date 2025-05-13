
"""
Morse Code Mapping and Alphabet Definitions

This module defines:
- A dictionary mapping alphanumeric characters and some punctuation to their Morse code representations.
- The full character alphabet used for model training or decoding.
- Index-based tag mapping for model outputs (useful in classification or sequence decoding).

The ALPHABET includes a leading space (" ") to account for blank/padding tokens often used in CTC decoding
or to represent word separation in inferred text.
"""

# A dictionary mapping each uppercase letter, digit, and selected punctuation to its Morse code representation
MORSE_CODE_DICT = {
    'A': '.-',    'B': '-...',  'C': '-.-.', 
    'D': '-..',   'E': '.',     'F': '..-.',
    'G': '--.',   'H': '....',  'I': '..',
    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',
    'S': '...',   'T': '-',     'U': '..-',
    'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',
    '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....',
    '7': '--...', '8': '---..', '9': '----.',
    '0': '-----',
    '.': '.-.-.-', ',': '--..--', '?': '..--..',
    '=': '-...-',  '+': '.-.-.',
}

# A string of all characters used for training or decoding, prefixed with a space character
# The space can represent a blank token (e.g., for CTC decoding) or a word separator
ALPHABET = " " + "".join(MORSE_CODE_DICT.keys())

# Dictionary mapping class indices to corresponding characters (1-based indexing)
# Useful for decoding model predictions (e.g., converting model outputs back into readable text)
num_tags = len(ALPHABET)
idx_to_tag = {i + 1: c for i, c in enumerate(ALPHABET)}

# Russian Morse code equivalent (based on the Russian version of Morse code)
RUSSIAN_MORSE_DICT = {
    'А': '.-', 'Б': '-...', 'В': '.--', 'Г': '--.', 'Д': '-..', 'Е': '.', 'Ё': '.', 'Ж': '...-', 'З': '--..',
    'И': '..', 'Й': '.---', 'К': '-.-', 'Л': '.-..', 'М': '--', 'Н': '-.', 'О': '---', 'П': '.--.',
    'Р': '.-.', 'С': '...', 'Т': '-', 'У': '..-', 'Ф': '..-.', 'Х': '....', 'Ц': '-.-.', 'Ч': '---.',
    'Ш': '----', 'Щ': '--.-', 'Ь': '-..-', 'Ы': '-.--', 'Э': '..-..', 'Ю': '..--', 'Я': '.-.-', '1': '.----',
    '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.', '0': '-----', '.': '.-.-.-', ',': '--..--', '?': '..--..', '=': '-...-', '+': '.-.-.'
}

# Function to convert from Latin Morse code to Russian Morse code
def latin_to_morse(latin_str):
    return ''.join(
        MORSE_CODE_DICT.get(char.upper(), '') + ' ' if char != ' ' else '_'
        for char in latin_str
    ).strip()

# Function to decode Russian Morse code into Cyrillic text
def morse_to_cyrillic_text(morse_code_str):
    morse_to_russian = {v: k for k, v in RUSSIAN_MORSE_DICT.items()}
    words = morse_code_str.strip().split('_')
    return ' '.join(
        ''.join(morse_to_russian.get(letter, '?') for letter in word.split())
        for word in words
    )
