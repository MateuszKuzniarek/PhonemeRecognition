import pickle
from pathlib import Path

from numpy import argmax, argwhere, delete, array
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.models import load_model

from src.constants import PHONEME_SYMBOLS
from src.data_preparation import get_features, prepare_training_data, prepare_testing_data
from src.dataset_explorer import DatasetExplorer
from src.model_creation import get_lstm_model, get_bidirectional_lstm_model, get_gru_model, get_bidirectional_gru_model, \
    get_peephole_lstm_model, get_bidirectional_peephole_lstm_model
from src.model_evaluation import show_test_results_for_models, train_model
from src.settings import NUMBER_OF_MFCCS, BATCH_SIZE, NUMBER_OF_EPOCHS, VALIDATION_SPLIT, DATASET_PATH, LOGS_PATH

if __name__ == "__main__":
    train_model(get_lstm_model, 'lstm')





    model_names = ['test_40_lstm_peephole_bidirectional', 'test_40_lstm_peephole', 'test_40_lstm',
                   'test_40_gru_bidirectional', 'test_40_gru', 'test_40_bidirectional']
    show_test_results_for_models(model_names)

