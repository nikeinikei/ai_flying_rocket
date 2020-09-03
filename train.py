import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import IMAGES_FILE_NAME, INPUTS_FILE_NAME


CHECKPOINTS_PATH = "saved/cp.ckpt"
CHECKPOINTS_DIR = os.path.dirname(CHECKPOINTS_PATH)


def main():
    images = tf.convert_to_tensor(np.load(IMAGES_FILE_NAME), dtype=float)
    inputs = tf.convert_to_tensor(np.load(INPUTS_FILE_NAME), dtype=float)

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        loss=tf.keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam()
    )

    latest = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
    if latest:
        print("continuing from checkpoint")
        model.load_weights(latest)

    print("using tensorflow version: ", tf.version.VERSION)

    print("images.shape", images.shape)
    print("inputs.shape", inputs.shape)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINTS_PATH,
        save_weights_only=True,
        verbose=1)

    try:
        history: tf.keras.callbacks.History = model.fit(
            x=images, 
            y=inputs,
            validation_split=0.2,
            epochs=300,
            callbacks=[cp_callback]
        )
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    except KeyboardInterrupt:
        print("manually interrupted")
        return


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        
    main()
