from src.data_preparation import get_features, prepare_data
from src.dataset_explorer import DatasetExplorer

if __name__ == "__main__":
    print('hello world')
    dataset_explorer = DatasetExplorer('../data/TIMIT')
    # for accent in dataset_explorer.get_accent_ids(True):
    #     print(accent)
    #     for speaker in dataset_explorer.get_speaker_ids(True, accent):
    #         print('--' + speaker)
    #         for sentence in dataset_explorer.get_sentences_ids(True, accent, speaker):
    #             print('----' + sentence)


    # wave_data = dataset_explorer.get_wave_data(True, 'DR1', 'FCJF0', 'SA1')
    # print(wave_data.raw_data)
    # print(len(wave_data.raw_data))
    # print(wave_data.sampling_rate)
    # features = get_features(wave_data)
    # print(features.shape)
    # phoneme_data = dataset_explorer.get_phoneme_data(True, 'DR1', 'FCJF0', 'SA1')
    # for row in phoneme_data:
    #     print(row.beginning)
    #     print(row.end)
    #     print(row.phoneme)

    x, y = prepare_data(True)