"""
Author: Dikshant Gupta
Time: 27.07.21 21:43
"""

import socket


class Connector:
    def __init__(self, port):
        self.port = port
        self.connection = None

    def establish_connection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', self.port)
        sock.bind(server_address)
        sock.listen()
        print("Connect to port {}".format(self.port))
        self.connection, client_address = sock.accept()
        print("Connection to port {} established!".format(self.port))

    def receive_message(self):
        message = ""
        while True:
            m = self.connection.recv(1024).decode('utf-8')
            # print("Received {}!".format(m))
            message += m
            if m[-1] == "\n":
                break
        return message

    def send_message(self, terminal, reward, angle, car_pos, car_speed,
                     pedestrian_positions, path, pedestrian_path=None):
        message = ""
        if terminal:
            temp = "true"
        else:
            temp = "false"
        message += temp + ";" + str(reward) + ";" + str(angle) + ";"
        message += str(car_pos[0]) + ";" + str(car_pos[1]) + ";" + str(car_speed) + ";"
        for pos in pedestrian_positions:
            if len(pos) == 0:
                message += ";" + ";"
            else:
                message += str(pos[0]) + ";" + str(pos[1]) + ";"  # Pedestrian position: (x, y)
        for wp in path:
            message += str(wp[0]) + "," + str(wp[1]) + "," + str(wp[2]) + ","  # Waypoint: (x, y, theta)

        message = message[:-1] + ";"
        if pedestrian_path is not None:
            for pos in pedestrian_path:
                message += str(pos[0]) + "," + str(pos[1]) + ","
        else:
            message += "null,"
        message = message[:-1] + "\n"
        message = message.encode('utf-8')
        self.connection.sendall(message)


if __name__ == '__main__':
    conn = Connector(1245)
    conn.establish_connection()
    conn.receive_message()
    for _ in range(10):
        conn.send_message(False, 0.0, 0.0, [100.0, 100.0], 8.5, [[89.0, 92.0]], [[101.0, 101.0, 0.0],
                                                                                 [102.0, 102.0, 0.0],
                                                                                 [103.0, 103.0, 0.0]])
        msg = conn.receive_message()
        print(msg[0] == '0', msg[0])
