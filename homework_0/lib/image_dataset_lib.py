import os
import pathlib
from typing import Tuple, Optional, List, Sequence
import tensorflow as tf
from numpy import array


def image_dataset_from_directory(
    dataset_path: pathlib.Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
) -> tf.data.Dataset:
    image_paths, labels, class_names = retrieve_directory_metadata(dataset_path)
    if shuffle:
        image_paths, labels, class_names = zip_shuffle_sequences([image_paths, labels, class_names], seed=seed)

    full_dataset = build_dataset(image_paths=image_paths, image_size=image_size, labels=labels)

    if subset == "both":
        if validation_split > 1:
            raise Exception("'validation split ratio' should be <= 1")
        train_dataset_size = int(validation_split * len(image_paths))

        train_dataset = full_dataset.take(train_dataset_size)
        val_dataset = full_dataset.skip(train_dataset_size)

        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)

        train_dataset.class_names = class_names[:train_dataset_size]
        val_dataset.class_names = class_names[train_dataset_size:]

        train_dataset.file_paths = image_paths[:train_dataset_size]
        val_dataset.file_paths = image_paths[train_dataset_size:]

        dataset = [train_dataset, val_dataset]
    else:
        dataset = full_dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset.class_names = class_names
        dataset.file_paths = image_paths
    return dataset


def zip_shuffle_sequences(sequences: List[Sequence], seed: Optional[int] = None) -> List[list]:
    if not sequences:
        raise Exception("sequences shouldn't be empty")
    indexes = [idx for idx in range(len(sequences[0]))]
    shuffled_indexes = tf.random.shuffle(indexes, seed=seed).numpy().tolist()
    return [array(seq)[shuffled_indexes].tolist() for seq in sequences]


def build_dataset(image_paths: List[str], image_size: Tuple[int, int], labels: List[int]) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    img_ds = path_ds.map(lambda x: load_image(x, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    return tf.data.Dataset.zip((img_ds, label_ds))


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
