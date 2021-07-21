from src.DatasetExplorer import DatasetExplorer

if __name__ == "__main__":
    print('hello world')
    dataset_explorer = DatasetExplorer('../data/TIMIT')
    wave_data = dataset_explorer.get_wave_data(True, 'DR1', 'FCJF0', 'SA1')
    print(wave_data.raw_data)
    print(wave_data.sampling_rate)
    phoneme_data = dataset_explorer.get_phoneme_data(True, 'DR1', 'FCJF0', 'SA1')
    for row in phoneme_data:
        print(row.beginning)
        print(row.end)
        print(row.phoneme)
