import tensorflow as tf
import numpy as np
from config import IMAGES_FILE_NAME
from train import create_model
import time


def main():
    model = create_model()

    images = np.load(IMAGES_FILE_NAME)
    model(tf.convert_to_tensor(np.array([images[0]]), dtype=float))

    start = time.perf_counter()

    for i in range(20):
        model(tf.convert_to_tensor(np.array([images[i]]), dtype=float))

    end = time.perf_counter()
    print("elapsed", end - start)


if __name__ == "__main__":
    main()