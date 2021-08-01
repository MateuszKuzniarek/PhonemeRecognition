from tensorflow.python.keras.layers import Masking, LSTM, TimeDistributed, Dense
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

from src.constants import PHONEME_SYMBOLS
from src.settings import LSTM_LAYER_SIZE


def get_lstm_model(input_shape):
    model = Sequential()
    model.add(Masking())
    model.add(LSTM(LSTM_LAYER_SIZE, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(LSTM_LAYER_SIZE, return_sequences=True))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model
