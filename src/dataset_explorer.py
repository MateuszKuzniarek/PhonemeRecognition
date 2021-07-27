import csv
import os

import librosa
import librosa.display

from src.data_formats import WaveFileData, PhonemeData


class DatasetExplorer:

    def __init__(self, dataset_directory: str):
        self.dataset_directory = dataset_directory

    def __create_accents_directory_path(self, train_set: bool):
        file_path = self.dataset_directory
        if train_set:
            file_path = file_path + '/' + 'TRAIN'
        else:
            file_path = file_path + '/' + 'TEST'
        return file_path

    def __create_speakers_directory_path(self, train_set: bool, accent_id: str) -> str:
        return self.__create_accents_directory_path(train_set) + '/' + accent_id

    def __create_sentences_directory_path(self, train_set: bool, accent_id: str, speaker_id: str) -> str:
        return self.__create_speakers_directory_path(train_set, accent_id) + '/' + speaker_id

    def __create_file_path(self, train_set: bool, accent_id: str, speaker_id: str, sentence_id: str) -> str:
        file_path = self.__create_sentences_directory_path(train_set, accent_id, speaker_id)
        file_path = file_path + '/' + sentence_id
        return file_path

    def get_wave_data(self, train_set: bool, accent_id: str, speaker_id: str, sentence_id: str) -> WaveFileData:
        file_path = self.__create_file_path(train_set, accent_id, speaker_id, sentence_id)
        file_path = file_path + '.wav'
        wave_data, sampling_rate = librosa.load(file_path)
        return WaveFileData(wave_data, sampling_rate)

    def get_phoneme_data(self, train_set: bool, accent_id: str, speaker_id: str, sentence_id: str) -> PhonemeData:
        file_path = self.__create_file_path(train_set, accent_id, speaker_id, sentence_id)
        file_path = file_path + '.phn'
        phoneme_series = []
        with open(file_path, newline='') as phoneme_file:
            reader = csv.reader(phoneme_file, delimiter=' ')
            for line in reader:
                phoneme_series.append(PhonemeData(int(line[0]), int(line[1]), line[2]))
        return phoneme_series

    def get_accent_ids(self, train_set: bool) -> [str]:
        accents_directory_path = self.__create_accents_directory_path(train_set)
        return [accent_id for accent_id in os.listdir(accents_directory_path)
                if os.path.isdir(accents_directory_path + '/' + accent_id)]

    def get_speaker_ids(self, train_set: bool, accent_id: str) -> [str]:
        speakers_directory_path = self.__create_speakers_directory_path(train_set, accent_id)
        return [speaker_id for speaker_id in os.listdir(speakers_directory_path)
                if os.path.isdir(speakers_directory_path + '/' + speaker_id)]

    def get_sentences_ids(self, train_set: bool, accent_id: str, speaker_id: str) -> [str]:
        sentences_directory_path = self.__create_sentences_directory_path(train_set, accent_id, speaker_id)
        file_names = [sentence_id.rsplit('.', 1)[0] for sentence_id in os.listdir(sentences_directory_path)
                      if not os.path.isdir(sentences_directory_path + '/' + sentence_id)]
        return list(set(file_names))

