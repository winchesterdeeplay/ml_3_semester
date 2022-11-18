import os
import tensorflow as tf
from typing import Tuple, Optional, List
import pathlib


class ImageDatasetLoader:
    def __init__(self):
        pass

    @classmethod
    def image_dataset_from_directory(
        cls,
        dataset_path: pathlib.Path,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (256, 256),
        shuffle: bool = True,
        seed: Optional[int] = None,
        validation_split: Optional[float] = None,
        subset: Optional[str] = None,
    ) -> tf.data.Dataset:

        image_paths, labels, class_names = ImageDatasetLoader.retrieve_directory_metadata(dataset_path)
        full_dataset = ImageDatasetLoader.build_dataset(image_paths=image_paths, image_size=image_size, labels=labels)
        if shuffle:
            full_dataset = full_dataset.shuffle(batch_size * 8, seed=42 if seed is None else seed)

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

    @classmethod
    def build_dataset(cls, image_paths: List[str], image_size: Tuple[int, int], labels: List[int]) -> tf.data.Dataset:
        path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        img_ds = path_ds.map(
            lambda x: ImageDatasetLoader.load_image(x, image_size), num_parallel_calls=tf.data.AUTOTUNE
        )
        return tf.data.Dataset.zip((img_ds, label_ds))

    @classmethod
    def retrieve_directory_metadata(cls, directory: pathlib.Path):
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

    @classmethod
    def load_image(cls, path, image_size, num_channels=3):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=num_channels, expand_animations=False)
        img = tf.image.resize(img, image_size, method="bilinear")
        img.set_shape((image_size[0], image_size[1], num_channels))
        return img
