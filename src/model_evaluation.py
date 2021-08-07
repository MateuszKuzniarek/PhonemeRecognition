import pickle
from pathlib import Path
from shutil import copyfile
from typing import Callable

from numpy import argmax, array
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.models import load_model, Model

from src.constants import PHONEME_SYMBOLS, SETTINGS_PATH
from src.data_preparation import prepare_training_data, prepare_testing_data
from src.normalization import Normalizer
from src.settings import LOGS_PATH, NUMBER_OF_MFCCS, BATCH_SIZE, NUMBER_OF_EPOCHS, VALIDATION_SPLIT


def save_results(normalizer: Normalizer, test_results: [], model_name: str):
    normalizer_name = open(LOGS_PATH + model_name + '/normalizer.pickle', "wb")
    pickle.dump(normalizer, normalizer_name)
    normalizer_name.close()
    with open(LOGS_PATH + model_name + '/results.txt', 'w') as result_file:
        result_file.write(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')
    copyfile(SETTINGS_PATH, LOGS_PATH + model_name + '/settings.py')


def train_model(get_model: Callable, model_name: str):
    Path(LOGS_PATH + model_name).mkdir(parents=True, exist_ok=True)
    x_train, y_train, normalizer = prepare_training_data()
    model = get_model(input_shape=(x_train.shape[1], NUMBER_OF_MFCCS * 3))
    csv_logger = CSVLogger(LOGS_PATH + model_name + '/logs.csv', append=True, separator=';')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS,
                        validation_split=VALIDATION_SPLIT, callbacks=[csv_logger])
    model.save(LOGS_PATH + model_name + '/model.h5', save_format='h5')
    x_test, y_test = prepare_testing_data(normalizer)
    test_results = model.evaluate(x_test, y_test, verbose=False)
    save_results(normalizer, test_results, model_name)
    show_example_model_response(model, x_test, y_test)


def show_example_model_response(model: Model, x_test: [], y_test: []):
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


def show_test_results_for_models(model_names: [str]):
    x_train, y_train, normalizer = prepare_training_data()
    x_test, y_test = prepare_testing_data(normalizer)
    for model_name in model_names:
        model = load_model('../logs/' + model_name + '/model.h5')
        test_results = model.evaluate(x_test, y_test, verbose=False)
        print('---' + model_name)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')