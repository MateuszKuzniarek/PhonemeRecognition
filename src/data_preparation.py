from keras_preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, delta
from numpy import concatenate, array, ndarray
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.constants import SILENCE_SYMBOL, PHONEME_SYMBOLS
from src.data_formats import WaveFileData, PhonemeData
from src.dataset_explorer import DatasetExplorer
from src.normalization import Normalizer
from src.settings import DATASET_PATH, NUMBER_OF_MFCCS


def get_features(wave_data: WaveFileData) -> [[float]]:
    mfccs = mfcc(y=wave_data.raw_data, sr=wave_data.sampling_rate, hop_length=wave_data.frame_length,
                 n_mfcc=NUMBER_OF_MFCCS)
    mfcc_deltas = delta(mfccs)
    mfcc_deltas_deltas = delta(mfccs, order=2)
    features = concatenate((mfccs, mfcc_deltas))
    features = concatenate((features, mfcc_deltas_deltas))
    return features.T


def get_phonemes_frame_vector(phoneme_series: [PhonemeData], frame_length: int, frame_count: int) -> [str]:
    result = []
    frame_fragments = []
    for phoneme_data in phoneme_series:
        phoneme_length = phoneme_data.end - phoneme_data.beginning
        fragments_length_left = frame_length - sum(map(lambda x: int(x[0]), frame_fragments))
        new_fragment_length = min(phoneme_length, fragments_length_left)
        frame_fragments.append((new_fragment_length, phoneme_data.phoneme))
        if new_fragment_length == fragments_length_left:
            phoneme_length = phoneme_length - new_fragment_length
            biggest_fragment = max(frame_fragments)
            result.append(biggest_fragment[1])
            frame_fragments.clear()
            number_of_full_frames = phoneme_length // frame_length
            frame_fragments.append((phoneme_length % frame_length, phoneme_data.phoneme))
            result.extend([phoneme_data.phoneme for i in range(number_of_full_frames)])
    biggest_fragment = max(frame_fragments)
    result.append(biggest_fragment[1])
    additional_frames_count = frame_count - len(result)
    result.extend([PHONEME_SYMBOLS.index(SILENCE_SYMBOL) for i in range(additional_frames_count)])
    result = result[:frame_count]
    return result


def construct_arrays(training_data: bool) -> ndarray:
    inputs = []
    outputs = []
    dataset_explorer = DatasetExplorer(DATASET_PATH)
    for accent in dataset_explorer.get_accent_ids(training_data):
        for speaker in dataset_explorer.get_speaker_ids(training_data, accent):
            for sentence in dataset_explorer.get_sentences_ids(training_data, accent, speaker):
                wave_data = dataset_explorer.get_wave_data(training_data, accent, speaker, sentence)
                phoneme_data = dataset_explorer.get_phoneme_data(training_data, accent, speaker, sentence)
                features = get_features(wave_data)
                inputs.append(features)
                output = get_phonemes_frame_vector(phoneme_data, wave_data.frame_length, features.shape[0])
                outputs.append(to_categorical(output, num_classes=len(PHONEME_SYMBOLS)))
                print(accent + speaker + sentence)
    return array(inputs), array(outputs)


def pad_arrays(inputs: ndarray, outputs: ndarray) -> ndarray:
    inputs = pad_sequences(inputs, dtype='float32', padding='post')
    outputs = pad_sequences(outputs, maxlen=inputs.shape[1], dtype='int32', padding='post')
    return inputs, outputs


def normalize_inputs(inputs: ndarray, normalizer: Normalizer):
    for sample in range(0, len(inputs)):
        for sequence in range(0, len(inputs[sample])):
            inputs[sample][sequence] = normalizer.normalize_vector(inputs[sample][sequence])


def prepare_training_data() -> (ndarray, ndarray, Normalizer):
    inputs, outputs = construct_arrays(training_data=True)
    normalizer = Normalizer(concatenate(inputs, axis=0))
    normalize_inputs(inputs, normalizer)
    inputs, outputs = pad_arrays(inputs, outputs)
    return inputs, outputs, normalizer


def prepare_testing_data(normalizer: Normalizer) -> (ndarray, ndarray):
    inputs, outputs = construct_arrays(training_data=False)
    normalize_inputs(inputs, normalizer)
    return pad_arrays(inputs, outputs)
