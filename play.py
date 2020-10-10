import socket
import json
import sys
import io
from train import create_model
import tensorflow as tf
import numpy as np
from PIL import Image
from config import IMAGES_FILE_NAME

TCP_IP = "127.0.0.1"
TCP_PORT = 5006


class ConnectionHandler(object):
    def __init__(self, conn, address, model):
        self.conn = conn
        self.address = address
        self.model = model
        self.delimiter = b'\n'
        self.delimiter_length = len(self.delimiter)
        self.bytes = bytearray()
        self.images = None

    def receive_until_delimiter(self):
        while True:
            try:
                index = self.bytes.index(self.delimiter)
                data = self.bytes[0:index]
                self.bytes = self.bytes[(index+1):]     # +1 to skip the delimiter, since it's not supposed to carry any data with it
                return data
            except ValueError:
                self.bytes += self.conn.recv(8192)
                continue

    def receive_fixed_amount(self, length: int):
        while len(self.bytes) < length:
            self.bytes += self.conn.recv(8192)
        fixed_amount = self.bytes[0:length]
        self.bytes = self.bytes[length:]
        return fixed_amount

    def add_image(self, image: np.ndarray):
        if self.images is None:
            self.images = image
        else:
            self.images = np.append(self.images, image, axis=0)

    def pedal_decision(self, x: float) -> float:
        if x >= 0.5:
            return 1.0
        else:
            return 0.0

    def rotation_decision(self, x: float) -> float:
        if x >= 0.35:
            return 1.0
        elif x <= -0.35:
            return -1.0
        else:
            return 0.0

    def send_prediction(self):
        prediction = self.model(tf.convert_to_tensor(np.array([self.images]), dtype=float))
        last_prediction = prediction[0][-1]
        rotation = float(last_prediction[0])
        pedal = float(last_prediction[1])
        print("rotation, pedal", rotation, pedal)
        game_input = {
            "rotation": self.rotation_decision(rotation),
            "pedal": self.pedal_decision(pedal)
        }
        self.conn.send(bytes(json.dumps(game_input) + "\n", "utf-8"))

    def run(self):
        while True:
            size_or_end = self.receive_until_delimiter()
            size_or_end = json.loads(size_or_end)
            if type(size_or_end) == str:
                if size_or_end == "end":
                    break
                else:
                    sys.exit("unknown string")

            image_data = self.receive_fixed_amount(int(size_or_end))
            image = Image.open(io.BytesIO(image_data))
            np_image = np.array(image)
            np_image = np_image.reshape((-1,))
            self.add_image(np.array([np_image]))
            self.send_prediction()

        self.conn.close()


def main():
    model = create_model()
    images = np.load(IMAGES_FILE_NAME)
    model(tf.convert_to_tensor(np.array([images[0]]), dtype=float))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))

    sock.listen(1)

    conn, address = sock.accept()

    connectionHandler = ConnectionHandler(conn, address, model)
    connectionHandler.run()

    sock.close()


if __name__ == "__main__":
    main()