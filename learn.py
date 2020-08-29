import numpy as np
import tensorflow as tf


def main():
    images = np.load("images.npy")
    (time_steps, _, _, _) = images.shape
    images = images.reshape((time_steps, -1))
    inputs = np.load("inputs.npy")

    batch_images = tf.convert_to_tensor(np.array([images]), dtype=float)
    batch_inputs = tf.convert_to_tensor(np.array([inputs]), dtype=float)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, return_sequences=True)),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"]
    )

    model.fit(x=batch_images, y=batch_inputs, epochs=300)


if __name__ == '__main__':
    main()
