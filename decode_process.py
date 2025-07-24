
import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.stats import norm

MORSE_CODE_EN_DICT = {
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

MORSE_CODE_RU_DICT = {
    'A': '.-',     'Б': '-...',   'В': '.--',
    'Г': '--.',    'Д': '-..',    'E': '.',
    'Ж': '...-',   'З': '--..',   'И': '..',
    'Й': '.---',   'K': '-.-',    'Л': '.-..',
    'M': '--',     'Н': '-.',     'O': '---',
    'П': '.--.',   'Р': '.-.',    'С': '...',
    'T': '-',      'У': '..-',    'Ф': '..-.',
    'X': '....',   'Ц': '-.-.',   'Ч': '---.',
    'Ш': '----',   'Щ': '--.-',   'Ъ': '--.--',
    'Ы': '-.--',   'Ь': '-..-',   'Э': '..-..',
    'Ю': '..--',   'Я': '.-.-',
    '1': '.----',  '2': '..---',  '3': '...--',
    '4': '....-',  '5': '.....',  '6': '-....',
    '7': '--...',  '8': '---..',  '9': '----.',
    '0': '-----',
    '.': '......', ',': '.-.-.-', '?': '..--..',
    '-': '-....-', '/': '-..-.',  '=': '-...-',
    '(': '-.--.',  ')': '-.--.-', '+': '.-.-.',
}

MORSE_TO_TEXT = {v: k for k, v in MORSE_CODE_EN_DICT.items()}

def get_segments(signal):
    segments = []
    if len(signal) == 0:
        return segments
    
    current = signal[0]
    count = 1
    for bit in signal[1:]:
        if bit == current:
            count += 1
        else:
            segments.append((current, count))
            current = bit
            count = 1
    segments.append((current, count))
    return segments

def cluster_durations(segments, tone=True, confidence_threshold=0.6):
    durations = np.array([dur for val, dur in segments if val == int(tone)]).reshape(-1, 1)
    if len(durations) < 2:
        return durations.flatten(), np.zeros_like(durations.flatten()), [0], [1]  # dummy std
    
    k = 2 if tone else 3
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(durations)
    labels = gmm.predict(durations)

    # Sort centers and stds together
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    sorted_indices = np.argsort(means)
    centers = means[sorted_indices]
    stds = stds[sorted_indices]

    # Filter by confidence
    probs = gmm.predict_proba(durations)
    mask = probs.max(axis=1) >= confidence_threshold
    durations = durations[mask]
    labels = labels[mask]

    return durations.flatten(), labels, centers, stds

def viterbi(observations, states, start_prob, trans_prob, emission_models, segment_types):
    V = [{}]
    path = {}

    for s in states:
        if valid_state(s, segment_types[0]):
            V[0][s] = np.log(start_prob[s] + 1e-10) + np.log(emission_models[s].pdf(observations[0]) + 1e-10)
            path[s] = [s]

    for t in range(1, len(observations)):
        V.append({})
        new_path = {}

        for curr in states:
            if not valid_state(curr, segment_types[t]):
                continue

            max_prob, best_prev = max(
                (
                    V[t-1][prev] + np.log(trans_prob[prev].get(curr, 1e-10)) +
                    np.log(emission_models[curr].pdf(observations[t]) + 1e-10),
                    prev
                )
                for prev in V[t-1]
            )
            V[t][curr] = max_prob
            new_path[curr] = path[best_prev] + [curr]

        path = new_path

    final_state = max(V[-1], key=V[-1].get)
    return path[final_state]

def segments_to_morse(labeled):
    morse = ""
    current_symbol = ""
    for label in labeled:
        if label == "dot":
            current_symbol += "."
        elif label == "dash":
            current_symbol += "-"
        elif label == "intra_gap":
            continue
        elif label == "char_gap":
            morse += current_symbol + " "
            current_symbol = ""
        elif label == "word_gap":
            morse += current_symbol + " / "
            current_symbol = ""
    morse += current_symbol
    return morse.strip()

def morse_to_text(morse):
    words = morse.split(" / ")
    decoded = []
    for word in words:
        letters = word.split()
        decoded_word = ''.join(MORSE_TO_TEXT.get(letter, '?') for letter in letters)
        decoded.append(decoded_word)
    return ' '.join(decoded)

def valid_state(state, segment_type):
    if segment_type == 1:
        return state in ['dot', 'dash']
    else:
        return state in ['intra_gap', 'char_gap', 'word_gap']

states = ['dot', 'dash', 'intra_gap', 'char_gap', 'word_gap']
start_prob = {s: 1/len(states) for s in states}

trans_prob = {
    'dot':       {'intra_gap': 0.80, 'char_gap': 0.15, 'word_gap': 0.15},
    'dash':      {'intra_gap': 0.80, 'char_gap': 0.15, 'word_gap': 0.15},
    'intra_gap': {'dot': 0.5, 'dash': 0.5},
    'char_gap':  {'dot': 0.5, 'dash': 0.5},
    'word_gap':  {'dot': 0.5, 'dash': 0.5},
}

def calculate_wpm(dot_duration_samples):
    dot_duration_sec = dot_duration_samples * 10
    if dot_duration_sec <= 0:
        return 0
    wpm = 1200 / dot_duration_sec
    return round(wpm, 2)

def convert2text(binary_signal, sample_rate=8000, lang='en'):

    global MORSE_TO_TEXT
    if lang == "ru":
        MORSE_TO_TEXT = {v: k for k, v in MORSE_CODE_RU_DICT.items()}
    else:
        MORSE_TO_TEXT = {v: k for k, v in MORSE_CODE_EN_DICT.items()}

    segments = get_segments(binary_signal)

    tone_durations, tone_labels, tone_centers, tone_stds = cluster_durations(segments, tone=True, confidence_threshold=0.8)
    gap_durations, gap_labels, gap_centers, gap_stds = cluster_durations(segments, tone=False, confidence_threshold=0.8)

    emission_models = {
        'dot': norm(loc=tone_centers[0], scale=tone_stds[0]),
        'dash': norm(loc=tone_centers[1], scale=tone_stds[1]),
        'intra_gap': norm(loc=gap_centers[0], scale=gap_stds[0]),
        'char_gap': norm(loc=gap_centers[1], scale=gap_stds[1]),
        'word_gap': norm(loc=gap_centers[2], scale=gap_stds[2]),
    }

    observations = np.array([dur for val, dur in segments])
    segment_types = np.array([val for val, dur in segments])
    
    labeled_segments = viterbi(
        observations,
        states,
        start_prob,
        trans_prob,
        emission_models,
        segment_types
    )

    
    morse = segments_to_morse(labeled_segments)
    decoded_text = morse_to_text(morse)
    
    chunk_wpm = calculate_wpm(dot_duration_samples=tone_centers[0])

    return decoded_text, morse, chunk_wpm
