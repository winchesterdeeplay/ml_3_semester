import os
import pathlib
from typing import Tuple, Optional, List, Sequence
import tensorflow as tf
from numpy import array


def image_dataset_from_directory(
    directory: pathlib.Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
) -> tf.data.Dataset:
    image_paths, labels, class_names = retrieve_directory_metadata(directory)
    if shuffle:
        image_paths, labels, class_names = zip_shuffle_sequences([image_paths, labels, class_names], seed=seed)

    (image_paths_train, labels_train, class_names_train), (
        image_paths_val,
        labels_val,
        class_names_val,
    ) = zip_train_validation_split_sequences([image_paths, labels, class_names], validation_split=validation_split)

    if subset in ("both", "training", "validation"):
        if validation_split is None:
            raise Exception("'validation split' should be specified")
        if validation_split >= 1:
            raise Exception("'validation split' should be < 1")

    if subset == "training":
        dataset = build_dataset(
            image_paths=image_paths_train,
            class_names=class_names_train,
            labels=labels_train,
            image_size=image_size,
            batch_size=batch_size,
        )
    elif subset == "validation":
        dataset = build_dataset(
            image_paths=image_paths_val,
            class_names=class_names_val,
            labels=labels_val,
            image_size=image_size,
            batch_size=batch_size,
        )
    elif subset == "both":
        train_dataset = build_dataset(
            image_paths=image_paths_train,
            class_names=class_names_train,
            labels=labels_train,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_dataset = build_dataset(
            image_paths=image_paths_val,
            class_names=class_names_val,
            labels=labels_val,
            image_size=image_size,
            batch_size=batch_size,
        )

        dataset = [train_dataset, val_dataset]
    else:
        dataset = build_dataset(
            image_paths=image_paths,
            class_names=class_names,
            image_size=image_size,
            labels=labels,
            batch_size=batch_size,
        )
    return dataset


def zip_shuffle_sequences(sequences: List[Sequence], seed: Optional[int] = None) -> List[list]:
    if not sequences:
        raise Exception("sequences shouldn't be empty")
    indexes = [idx for idx in range(len(sequences[0]))]
    shuffled_indexes = tf.random.shuffle(indexes, seed=seed).numpy().tolist()
    return [array(seq)[shuffled_indexes].tolist() for seq in sequences]


def zip_train_validation_split_sequences(
        sequences: List[Sequence], validation_split: Optional[float] = None
) -> Tuple[List[Sequence], List[Sequence]]:
    if validation_split is None:
        return sequences, [[] for _ in range(len(sequences))]
    if not sequences:
        raise Exception("sequences shouldn't be empty")

    train_dataset_size = int((1-validation_split) * len(sequences[0]))

    train_sequences, validation_sequences = [], []

    for seq in sequences:
        train_sequences.append(seq[:train_dataset_size])
        validation_sequences.append(seq[-train_dataset_size:])
    return train_sequences, validation_sequences


def build_dataset(
    image_paths: List[str], class_names: List[str], labels: List[int], image_size: Tuple[int, int], batch_size: int
) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    img_ds = path_ds.map(lambda x: load_image(x, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = tf.data.Dataset.zip((img_ds, label_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)
    dataset.class_names = class_names
    dataset.file_paths = image_paths
    return dataset


def retrieve_directory_metadata(directory: pathlib.Path) -> Tuple[List[str], List[int], List[str]]:
    os.chdir(directory)
    image_paths, labels, class_names = [], [], []
    for idx, paths in enumerate(os.walk(directory), start=-1):
        if idx == -1:  # skip root directory
            continue
        class_name = paths[0].split("/")[-1]
        for image_path in paths[2]:
            image_paths.append(os.path.join(paths[0], image_path))
            labels.append(idx)
            class_names.append(class_name)
    return image_paths, labels, class_names


def load_image(path, image_size, num_channels=3) -> tf.image:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img
