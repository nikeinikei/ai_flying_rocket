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

WRITE_TO_DISK = False


class LearningThread(Thread):
    def __init__(self, conn, address):
        Thread.__init__(self)
        self.conn = conn
        self.address = address
        self.delimiter = b'\n'
        self.delimiter_length = len(self.delimiter)
        self.bytes = bytearray()

    def receive_until_delimiter(self):
        while True:
            try:
                index = self.bytes.index(self.delimiter)
                data = self.bytes[0:index]
                self.bytes = self.bytes[(index+1):]
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

    def run(self):
        if os.path.exists(INPUTS_FILE_NAME) and os.path.exists(IMAGES_FILE_NAME):
            all_inputs = np.load(INPUTS_FILE_NAME)
            all_images = np.load(IMAGES_FILE_NAME)
        else:
            all_inputs = np.array([])
            all_images = np.array([])

        inputs = []
        images = []

        while True:
            data = self.receive_until_delimiter()
            game_input = json.loads(data)
            if game_input == "won":
                print("won")
                print("len(inputs)", len(inputs))
                print("len(images)", len(images))
                all_inputs = np.append(all_inputs, np.array(inputs))
                all_images = np.append(all_images, np.array(images))
                inputs.clear()
                images.clear()
                continue
            elif game_input == "lost":
                print("lost")
                inputs = []
                images = []
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
            images.append(np_image)

        if WRITE_TO_DISK:
            np.save("inputs.npy", all_inputs)
            np.save("images.npy", all_images)
        print("end of connection")
        self.conn.close()


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
