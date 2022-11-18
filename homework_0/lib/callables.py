import pathlib
from typing import Tuple, Any, List, Callable

import matplotlib.pyplot as plt
import tensorflow as tf

from homework_0.lib.image_dataset import ImageDatasetLoader


def load_dataset(url: str, filename: str) -> pathlib.Path:
    data_dir = tf.keras.utils.get_file(filename, origin=url, untar=True)
    return pathlib.Path(data_dir)


def visualize_history(history: Any, epochs: int) -> None:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


def prefetch_ds(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds


def init_simple_model(class_names: List[str], img_height: int, img_width: int) -> tf.keras.models.Sequential:
    num_classes = len(class_names)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )

    return model


def choose_image_dataset_from_directory_method(tf_implementation: bool) -> Callable:
    if tf_implementation:
        return tf.keras.utils.image_dataset_from_directory
    return ImageDatasetLoader.image_dataset_from_directory


def retrieve_train_val_dataset(
    mode: str,
    image_dataset_from_directory_method: Callable,
    dataset_path: pathlib.Path,
    validation_split_ratio: float,
    batch_size: int,
    seed: int,
    image_size: Tuple[int, int],
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    if mode != "both":
        train_ds = image_dataset_from_directory_method(
            dataset_path=dataset_path,
            validation_split=validation_split_ratio,
            batch_size=batch_size,
            subset="training",
            seed=seed,
            image_size=image_size,
        )

        val_ds = image_dataset_from_directory_method(
            dataset_path=dataset_path,
            validation_split=validation_split_ratio,
            batch_size=batch_size,
            subset="validation",
            seed=seed,
            image_size=image_size,
        )
        return train_ds, val_ds
    else:
        return image_dataset_from_directory_method(
            dataset_path=dataset_path,
            validation_split=validation_split_ratio,
            batch_size=batch_size,
            subset="both",
            seed=seed,
            image_size=image_size,
        )
