from src.constants import MILLISECONDS_IN_SECOND
from src.settings import FRAME_LENGTH_IN_MS


class PhonemeData:

    def __init__(self, beginning: int, end: int, phoneme: str):
        self.beginning = beginning
        self.end = end
        self.phoneme = phoneme


class WaveFileData:

    def __init__(self, raw_data: [], sampling_rate: int):
        self.raw_data = raw_data
        self.sampling_rate = sampling_rate
        frames_per_millisecond = sampling_rate / MILLISECONDS_IN_SECOND
        self.frame_length = int(frames_per_millisecond * FRAME_LENGTH_IN_MS)