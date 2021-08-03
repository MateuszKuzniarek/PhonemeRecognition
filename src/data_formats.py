from src.constants import MILLISECONDS_IN_SECOND, PHONEME_SYMBOLS, PHONEME_MAPPINGS, SILENCE_SYMBOL
from src.settings import FRAME_LENGTH_IN_MS


class PhonemeData:

    def __init__(self, beginning: int, end: int, phoneme: str):
        self.beginning = beginning
        self.end = end
        if phoneme in PHONEME_SYMBOLS:
            self.phoneme = PHONEME_SYMBOLS.index(phoneme)
        else:
            self.__find_mapping(phoneme)

    def __find_mapping(self, phoneme: str):
        mapping_found = False
        for key in PHONEME_MAPPINGS:
            for value in PHONEME_MAPPINGS[key]:
                if phoneme == value:
                    mapping_found = True
                    self.phoneme = PHONEME_SYMBOLS.index(key)
        if mapping_found is False:
            print('no mapping found for ' + phoneme)
            self.phoneme = PHONEME_SYMBOLS.index(SILENCE_SYMBOL)


class WaveFileData:

    def __init__(self, raw_data: [], sampling_rate: int):
        self.raw_data = raw_data
        self.sampling_rate = sampling_rate
        frames_per_millisecond = sampling_rate / MILLISECONDS_IN_SECOND
        self.frame_length = int(frames_per_millisecond * FRAME_LENGTH_IN_MS)