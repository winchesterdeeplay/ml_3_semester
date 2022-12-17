import math

from keras.utils import Sequence

from homework_0.lib.image_dataset_lib import zip_shuffle_sequences


class CustomSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, seed=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.seed = seed

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return batch_x, batch_y

    def on_epoch_end(self):
        self.x, self.y = zip_shuffle_sequences([self.x, self.y], seed=self.seed)
