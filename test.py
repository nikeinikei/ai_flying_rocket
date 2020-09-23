import tensorflow as tf
import numpy as np
from config import IMAGES_FILE_NAME
from train import create_model
import time


def main():
    model = create_model(train=False)

    images = np.load(IMAGES_FILE_NAME)
    model(tf.convert_to_tensor(np.array([images[0]]), dtype=float))

    start = time.perf_counter()

    for i in range(25):
        model(tf.convert_to_tensor(np.array([images[i]]), dtype=float))

    end = time.perf_counter()
    print("elapsed", end - start)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()