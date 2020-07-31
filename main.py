import socket
from threading import Thread
import json


TCP_IP = "127.0.0.1"
TCP_PORT = 5005


class LearningThread(Thread):
    def __init__(self, conn, addr):
        print("accepting new connection ", conn)
        Thread.__init__(self)
        self.conn = conn
        self.addr = addr

    def construct_move(self) -> bytes:
        return bytes(json.dumps({'rotation': 0, 'pedal': 1}), 'utf-8')

    def run(self):
        raw_level_data = self.conn.recv(32768)
        level_data = json.loads(raw_level_data.decode('utf-8'))
        print("level data", level_data)

        while True:
            move = self.construct_move()
            self.conn.sendall(move)
            self.conn.sendall(b'\r\n')

            data = self.conn.recv(32768)

            if not data:
                self.conn.close()
                break
            else:
                game_state = json.loads(data.decode('utf-8'))

        self.conn.close()
        print("end of connection")


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))

    sock.listen(1)

    print("listening")

    while True:
        conn, addr = sock.accept()

        learning_thread = LearningThread(conn, addr)
        learning_thread.start()
        learning_thread.join()

    sock.close()


if __name__=="__main__":
    main()