import socket
from threading import Thread
import json
import io
from PIL import Image
import numpy as np
import os
from config import INPUTS_FILE_NAME, IMAGES_FILE_NAME

TCP_IP = "127.0.0.1"
TCP_PORT = 5005

WRITE_TO_DISK = True


class LearningThread(Thread):
    def __init__(self, conn, address):
        Thread.__init__(self)
        self.conn = conn
        self.address = address
        self.delimiter = b'\n'
        self.delimiter_length = len(self.delimiter)
        self.bytes = bytearray()
        if os.path.exists(INPUTS_FILE_NAME) and os.path.exists(IMAGES_FILE_NAME):
            self.inputs = np.load(INPUTS_FILE_NAME)
            self.images = np.load(IMAGES_FILE_NAME)
        else:
            self.inputs = None
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

    def add_recording(self, inputs, images):
        if self.inputs is None and self.images is None:
            self.inputs = np.array(inputs)
            self.images = np.array(images)
        else:
            timesteps = self.inputs.shape[1]
            timesteps2 = inputs.shape[1]

            if timesteps > timesteps2:
                diff = timesteps - timesteps2
                inputs = np.pad(inputs, [(0, 0), (0, diff), (0, 0)])
                images = np.pad(images, [(0, 0), (0, diff), (0, 0)])
            else:
                diff = timesteps2 - timesteps
                self.inputs = np.pad(self.inputs, [(0, 0), (0, diff), (0, 0)])
                self.images = np.pad(self.images, [(0, 0), (0, diff), (0, 0)])
            
            self.inputs = np.append(self.inputs, inputs, axis=0)
            self.images = np.append(self.images, images, axis=0)

    def save_recordins(self):
        if not self.inputs is None and not self.images is None:
            np.save(INPUTS_FILE_NAME, self.inputs)
            np.save(IMAGES_FILE_NAME, self.images)

    def run(self):
        inputs = []
        images = []

        while True:
            data = self.receive_until_delimiter()
            game_input = json.loads(data)
            if game_input == "won":
                print("won")
                self.add_recording(
                    np.array([inputs]), 
                    np.array([images])
                )
                print("inputs.shape", self.inputs.shape)
                print("images.shape", self.images.shape)
                inputs.clear()
                images.clear()
                continue
            elif game_input == "lost":
                print("lost")
                inputs.clear()
                images.clear()
                continue
            elif game_input == "quit":
                print("quit")
                break

            np_game_input = np.array([game_input["rotation"], game_input["pedal"]])
            inputs.append(np_game_input)

            image_file_size = int(self.receive_until_delimiter())
            data = self.receive_fixed_amount(image_file_size)

            image = Image.open(io.BytesIO(data))
            np_image = np.array(image)
            np_image = np_image.reshape((-1,))
            images.append(np_image)

        if WRITE_TO_DISK:
            self.save_recordins()
        self.conn.close()
        print("end of connection")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))

    sock.listen(1)

    print("listening")

    while True:
        conn, address = sock.accept()

        learning_thread = LearningThread(conn, address)
        learning_thread.run()

        break

    sock.close()


if __name__ == "__main__":
    main()
