import logging

from keras import layers
from keras.datasets import mnist
from tensorflow import keras

from homework_0.lib.tf_sequence import CustomSequence

logger = logging.getLogger(__name__)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 2

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential(
        [
            layers.Input(shape=(784,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    train_sequence = CustomSequence(x_train, y_train, batch_size=BATCH_SIZE, seed=SEED)
    history = model.fit(train_sequence, epochs=EPOCHS)

    loss, accuracy = model.evaluate(x_test, y_test)

    logger.info(f"loss: {loss}")
    logger.info(f"accuracy: {accuracy}")
