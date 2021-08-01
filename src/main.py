from src.data_preparation import get_features, prepare_training_data, prepare_testing_data
from src.dataset_explorer import DatasetExplorer
from src.model_creation import get_lstm_model
from src.settings import NUMBER_OF_MFCCS, BATCH_SIZE, NUMBER_OF_EPOCHS, VALIDATION_SPLIT

if __name__ == "__main__":
    x_train, y_train, normalizer = prepare_training_data()
    model = get_lstm_model(input_shape=(x_train.shape[1], NUMBER_OF_MFCCS * 3))

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS,
                        validation_split=VALIDATION_SPLIT)

    x_test, y_test = prepare_testing_data(normalizer)

    test_results = model.evaluate(x_test, y_test, verbose=False)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')