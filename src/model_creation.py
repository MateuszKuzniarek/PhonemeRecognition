from tensorflow.python.keras.layers import Masking, LSTM, TimeDistributed, Dense, Bidirectional, GRU, PeepholeLSTMCell, \
    RNN
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
    model.add(LSTM(LSTM_LAYER_SIZE, return_sequences=True))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model


def get_gru_model(input_shape):
    model = Sequential()
    model.add(Masking())
    model.add(GRU(LSTM_LAYER_SIZE, input_shape=input_shape, return_sequences=True))
    model.add(GRU(LSTM_LAYER_SIZE, return_sequences=True))
    model.add(GRU(LSTM_LAYER_SIZE, return_sequences=True))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model


def get_bidirectional_lstm_model(input_shape):
    model = Sequential()
    model.add(Masking())
    model.add(Bidirectional(LSTM(LSTM_LAYER_SIZE, input_shape=input_shape, return_sequences=True)))
    model.add(Bidirectional(LSTM(LSTM_LAYER_SIZE, return_sequences=True)))
    model.add(Bidirectional(LSTM(LSTM_LAYER_SIZE, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model


def get_bidirectional_gru_model(input_shape):
    model = Sequential()
    model.add(Masking())
    model.add(Bidirectional(GRU(LSTM_LAYER_SIZE, input_shape=input_shape, return_sequences=True)))
    model.add(Bidirectional(GRU(LSTM_LAYER_SIZE, return_sequences=True)))
    model.add(Bidirectional(GRU(LSTM_LAYER_SIZE, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model


def get_peephole_lstm_model(input_shape):
    model = Sequential()
    model.add(Masking())
    model.add(RNN(PeepholeLSTMCell(LSTM_LAYER_SIZE), input_shape=input_shape, return_sequences=True))
    model.add(RNN(PeepholeLSTMCell(LSTM_LAYER_SIZE), return_sequences=True))
    model.add(RNN(PeepholeLSTMCell(LSTM_LAYER_SIZE), return_sequences=True))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model


def get_bidirectional_peephole_lstm_model(input_shape):
    model = Sequential()
    model.add(Masking())
    model.add(Bidirectional(RNN(PeepholeLSTMCell(LSTM_LAYER_SIZE), input_shape=input_shape, return_sequences=True)))
    model.add(Bidirectional(RNN(PeepholeLSTMCell(LSTM_LAYER_SIZE), return_sequences=True)))
    model.add(Bidirectional(RNN(PeepholeLSTMCell(LSTM_LAYER_SIZE), return_sequences=True)))
    model.add(TimeDistributed(Dense(len(PHONEME_SYMBOLS), activation='softmax')))
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model
