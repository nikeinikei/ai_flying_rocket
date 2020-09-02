import numpy as np
import tensorflow as tf
from config import IMAGES_FILE_NAME, INPUTS_FILE_NAME


def main():
    images = tf.convert_to_tensor(np.load(IMAGES_FILE_NAME), dtype=float)
    inputs = tf.convert_to_tensor(np.load(INPUTS_FILE_NAME), dtype=float)

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(4, return_sequences=True),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        loss=tf.keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.fit(x=images, y=inputs, epochs=10)


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        
    main()
