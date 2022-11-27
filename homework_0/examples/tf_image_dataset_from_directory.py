from homework_0.lib.callables import (
    load_dataset,
    visualize_history,
    init_simple_model,
    prefetch_ds,
    choose_image_dataset_from_directory_method,
    retrieve_train_val_dataset,
)

IMAGE_DATASET_FROM_DIRECTORY_TF_IMPLEMENTATION = False  # Change to True for choosing tf implementation.

URL_DATASET = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
FILENAME_DATASET = "flower_photos"

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
RANDOM_SEED = 42
VALIDATION_SPLIT_RATIO = 0.2

TRAIN_EPOCHS = 10
SUBSET_MODE = "both"  # or any


if __name__ == "__main__":
    dataset_path = load_dataset(URL_DATASET, FILENAME_DATASET)

    image_dataset_from_directory_method = choose_image_dataset_from_directory_method(
        IMAGE_DATASET_FROM_DIRECTORY_TF_IMPLEMENTATION
    )

    train_ds, val_ds = retrieve_train_val_dataset(
        mode=SUBSET_MODE,
        image_dataset_from_directory_method=image_dataset_from_directory_method,
        dataset_path=dataset_path,
        validation_split_ratio=VALIDATION_SPLIT_RATIO,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
    )
    class_names = train_ds.class_names
    train_ds, val_ds = prefetch_ds(train_ds, val_ds)

    model = init_simple_model(class_names, IMG_HEIGHT, IMG_WIDTH)

    history = model.fit(train_ds, validation_data=val_ds, epochs=TRAIN_EPOCHS)

    visualize_history(history, TRAIN_EPOCHS)
