import pickle
from pathlib import Path

from numpy import argmax, argwhere, delete, array
from tensorflow.python.keras.callbacks import CSVLogger

from src.constants import PHONEME_SYMBOLS
from src.data_preparation import get_features, prepare_training_data, prepare_testing_data
from src.dataset_explorer import DatasetExplorer
from src.model_creation import get_lstm_model, get_bidirectional_lstm_model
from src.settings import NUMBER_OF_MFCCS, BATCH_SIZE, NUMBER_OF_EPOCHS, VALIDATION_SPLIT, DATASET_PATH, LOGS_PATH

if __name__ == "__main__":
    # dataset_explorer = DatasetExplorer(DATASET_PATH)
    # phoneme_data = dataset_explorer.get_phoneme_data(train_set=True, accent_id='DR1', speaker_id='FCJF0',
    #                                                  sentence_id='SA1')
    # wav_data = dataset_explorer.get_wave_data(train_set=True, accent_id='DR1', speaker_id='FCJF0', sentence_id='SA1')
    # print(len(wav_data.raw_data))
    # print(phoneme_data[-1].end)
    # print(len(wav_data.raw_data) / wav_data.sampling_rate)

    model_name = 'test_40_bidirectional'
    Path(LOGS_PATH + model_name).mkdir(parents=True, exist_ok=True)
    x_train, y_train, normalizer = prepare_training_data()
    model = get_bidirectional_lstm_model(input_shape=(x_train.shape[1], NUMBER_OF_MFCCS * 3))
    csv_logger = CSVLogger(LOGS_PATH + model_name + '/logs.csv', append=True, separator=';')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS,
                        validation_split=VALIDATION_SPLIT, callbacks=[csv_logger])
    model.save(LOGS_PATH + model_name + '/model.h5', save_format='h5')
    x_test, y_test = prepare_testing_data(normalizer)

    normalizer_name = open(LOGS_PATH + model_name + '/normalizer.pickle', "wb")
    pickle.dump(normalizer, normalizer_name)
    normalizer_name.close()

    test_results = model.evaluate(x_test, y_test, verbose=False)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')

    y_test_symbols = []
    x_test_0 = x_test[0]
    x_test_0 = x_test_0[~(x_test_0 == 0).all(1)]
    for symbol in y_test[0]:
        if (symbol != 0).any():
            y_test_symbols.append(PHONEME_SYMBOLS[argmax(symbol)])
    print(y_test_symbols)
    test_predictions = model.predict(array([x_test_0]))
    predictions = []
    for symbol in test_predictions[0]:
        predictions.append(PHONEME_SYMBOLS[argmax(symbol)])
    print(predictions)
