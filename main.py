import socket
from threading import Thread
import json
import io
from PIL import Image
import numpy as np

TCP_IP = "127.0.0.1"
TCP_PORT = 5005

IMAGE_DIMS = (80, 80, 3)


class LearningThread(Thread):
    def __init__(self, conn, address):
        Thread.__init__(self)
        self.conn = conn
        self.address = address

    def run(self):
        inputs = []
        images = []

        while True:
            game_input = json.loads(self.conn.recv(4096))
            if game_input == 0:
                break
            np_game_input = np.array([game_input["rotation"], game_input["pedal"]])
            inputs.append(np_game_input)

            image_file_size = int(self.conn.recv(8192))
            data = bytearray()
            while len(data) < image_file_size:
                more_data = self.conn.recv(8192)
                data += more_data

            image = Image.open(io.BytesIO(data))
            (w, h, d) = IMAGE_DIMS
            image = image.resize((w, h), Image.BILINEAR)
            np_image = np.array(image)
            images.append(np_image)

        inputs = np.array(inputs)
        images = np.array(images)

        np.save("inputs.npy", inputs)
        np.save("images.npy", images)
        print("end of connection")
        self.conn.close()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))

    sock.listen(1)

    print("listening")

    while True:
        conn, address = sock.accept()

        learning_thread = LearningThread(conn, address,)
        learning_thread.run()

        break

    sock.close()


if __name__ == "__main__":
    main()
