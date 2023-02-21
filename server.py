#! venv/bin/python3
from __future__ import annotations
import re
import sys
import socket
import logging
import numpy as np
from enum import Enum
from threading import Thread


class Server:
    def __init__(self, address: str, port: int):
        print(address, port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((address, port))
        self.sock = sock

    def start(self):
        self.sock.listen(1024)
        logging.info('Server is listening.')

        while True:
            conn, remote = self.sock.accept()
            logging.info(f'New connection: {remote}')

            Thread(target=Communication(conn, remote).start).start()


class Rotation:
    UP = np.array((0, -1))
    RIGHT = np.array((1,  0))
    DOWN = np.array((0,  1))
    LEFT = np.array((-1,  0))

    @classmethod
    def turn_left(cls, current: Rotation):
        return np.array(((0, -1), (1, 0))) @ current

    @classmethod
    def turn_right(cls, current: Rotation):
        return np.array(((0, 1), (-1, 0))) @ current


def dist_from_centre(position: np.ndarray):
    return np.sum(np.abs(position))


def move_decorator(func):
    def inner(self, *args, **kwargs):
        pos = self.position
        rot = self.rotation
        stuck = self.stuck
        move = func(self, *args, **kwargs)
        logging.debug(f'Moving: {pos=}, {rot=}, {stuck=}, {move=}')
        return move
    return inner


def position_decorator(func):
    def inner(self, *args, **kwargs):
        pos = self.position
        ret = func(self, *args, **kwargs)
        new_pos = self.position
        rot = self.rotation
        stuck = self.stuck
        logging.debug(f'New position: {pos=}, {new_pos=} {rot=}, {stuck=}')
        return ret
    return inner


class Robot:
    SERVER_MOVE = b'102 MOVE\a\b'
    SERVER_TURN_LEFT = b'103 TURN LEFT\a\b'
    SERVER_TURN_RIGHT = b'104 TURN RIGHT\a\b'

    def __init__(self):
        self.rotation = None
        self.position = None
        self.stuck = False
        self.move = None

    @position_decorator
    def add_position(self, position: np.ndarray):
        if self.position is None:
            self.position = position
            return self._at_start

        if self.rotation is None:
            dp = position - self.position
            self.position = position
            if np.all(dp == (0, 0)):
                self.stuck = self.move == Robot.SERVER_MOVE
                return self._at_start

            self.stuck = False
            self.rotation = dp
            return self._at_start

        if np.all(position == self.position) and self.move == Robot.SERVER_MOVE:
            self.stuck = True
            return self._at_start

        if self.stuck and self.move in (Robot.SERVER_TURN_RIGHT, Robot.SERVER_TURN_LEFT):
            return self._at_start

        self.stuck = False
        self.position = position
        return self._at_start

    @property
    def _at_start(self):
        if self.position is None:
            return False
        return np.all(self.position == (0, 0))

    @move_decorator
    def get_move(self):
        if self.position is None:
            self.move = Robot.SERVER_TURN_RIGHT
            return self.move

        if self.rotation is None:
            if self.stuck:
                self.move = Robot.SERVER_TURN_LEFT
                return self.move
            self.move = Robot.SERVER_MOVE
            return self.move

        if self.stuck or dist_from_centre(self.position + self.rotation) >= dist_from_centre(self.position):
            if self.stuck and self.move in (Robot.SERVER_TURN_RIGHT, Robot.SERVER_TURN_LEFT):
                self.move = Robot.SERVER_MOVE
                return self.move
            return self._get_next_rotation()

        self.move = Robot.SERVER_MOVE
        return self.move

    def _get_next_rotation(self):
        next_rot = Rotation.turn_left(self.rotation)
        if dist_from_centre(self.position + next_rot) < dist_from_centre(self.position):
            self.rotation = next_rot
            self.move = Robot.SERVER_TURN_LEFT
            return self.move
        else:
            self.rotation = Rotation.turn_right(self.rotation)
            self.move = Robot.SERVER_TURN_RIGHT
            return self.move


class ServerLoginFailedError(Exception):
    ...


class ServerSyntaxError(Exception):
    ...


class ServerLogicError(Exception):
    ...


class ServerKeyOutOfRangeError(Exception):
    ...


class Stage(Enum):
    INITIAL = 0
    AWAITING_KEY_ID = 1
    AWAITING_CLIENT_HASH = 2
    MOVING = 3
    AWAITING_SECRET = 4


class Communication:
    _key_pairs = ((23019, 32037),
                  (32037, 29295),
                  (18789, 13603),
                  (16443, 29533),
                  (18189, 21952))
    MODULO = 65536

    def __init__(self, conn: socket.socket, addr):
        self.sock = conn
        self.addr = addr
        self.buffer = ''
        self.stage = Stage.INITIAL
        self.charging = False
        self.remote_name = None
        self.name_hash = None
        self.key_id = None
        self.key_pair = None
        self.robot = Robot()

    def start(self):
        while True:
            if self.charging:
                self.sock.settimeout(5)
            else:
                self.sock.settimeout(1)
            try:
                data = self.sock.recv(1024)
            except TimeoutError:
                return self._terminate()

            if not data:
                return self._terminate()

            logging.debug(f'Message from {self.addr}: {data}')
            stream = data.decode()

            if not self._parse_stream(stream):
                return

    def _parse_stream(self, stream: str):
        """ Return False if connection should be terminated. """
        self.buffer += stream
        try:
            if not self._validate_buffer():
                return False

            while (idx := self.buffer.find('\a\b')) != -1:
                message = self.buffer[:idx]
                self.buffer = self.buffer[idx+2:]
                if not self._parse_message(message):
                    return False
            return True

        except ServerLoginFailedError:
            self.sock.send(b'300 LOGIN FAILED\a\b')
            logging.debug('Sending: 300 LOGIN FAILED')
            self._terminate()
            return False

        except ServerSyntaxError:
            self.sock.send(b'301 SYNTAX ERROR\a\b')
            logging.debug('Sending: 301 SYNTAX ERROR')
            self._terminate()
            return False

        except ServerLogicError:
            self.sock.send(b'302 LOGIC ERROR\a\b')
            logging.debug('Sending: 302 LOGIC ERROR')
            self._terminate()
            return False

        except ServerKeyOutOfRangeError:
            self.sock.send(b'303 KEY OUT OF RANGE\a\b')
            logging.debug('Sending: 303 KEY OUT OF RANGE')
            self._terminate()
            return False

    def _parse_message(self, message: str):
        """ Return False if connection should be terminated. """
        if self._parse_charging(message):
            return True

        if self.stage == Stage.INITIAL:
            self._parse_name(message)
        elif self.stage == Stage.AWAITING_KEY_ID:
            self._parse_key_id(message)
        elif self.stage == Stage.AWAITING_CLIENT_HASH:
            self._parse_client_hash(message)
        elif self.stage == Stage.MOVING:
            position = self._parse_move(message)
            if self.robot.add_position(position):
                self.sock.send(b'105 GET MESSAGE\a\b')
                self.stage = Stage.AWAITING_SECRET
                return True
            move = self.robot.get_move()
            self.sock.send(move)

        elif self.stage == Stage.AWAITING_SECRET:
            self.sock.send(b'106 LOGOUT\a\b')
            self._terminate()
            return False
        return True

    def _parse_name(self, message: str):
        self.remote_name = message
        self.name_hash = sum(ord(c) for c in message) * 1000 % Communication.MODULO
        self.stage = Stage.AWAITING_KEY_ID
        self.sock.send(b'107 KEY REQUEST\a\b')
        logging.debug('Sending: 107 KEY REQUEST')
        # todo: check length max 18

    def _parse_key_id(self, message):
        try:
            self.key_id = int(message)
        except ValueError:
            raise ServerSyntaxError
        if self.key_id > 4:
            raise ServerKeyOutOfRangeError
        self.key_pair = self._key_pairs[self.key_id]
        server_hash = (self.key_pair[0] + self.name_hash) % Communication.MODULO
        self.sock.send(f'{server_hash}\a\b'.encode())
        logging.debug(f'Sending: {server_hash}')
        self.stage = Stage.AWAITING_CLIENT_HASH
        # todo: check length max 3

    def _parse_client_hash(self, message):
        m = re.match(r'^(\d{1,5})$', message)
        if not m:
            raise ServerSyntaxError

        message = int(m.group())
        client_hash = (self.key_pair[1] + self.name_hash) % Communication.MODULO
        if client_hash != message:
            raise ServerLoginFailedError

        self.sock.send(b'200 OK\a\b')
        logging.debug('Sending: 200 OK')

        self.stage = Stage.MOVING
        move = self.robot.get_move()
        self.sock.send(move)

    def _validate_buffer(self):
        if self.stage == Stage.INITIAL:
            if len(self.buffer) == 19:
                if self.buffer[-1] != '\a' and self.buffer.find('\a\b') == -1:
                    raise ServerSyntaxError
            if len(self.buffer) >= 20:
                text_len = self.buffer.find('\a\b')
                if text_len == -1 or text_len > 18:
                    raise ServerSyntaxError
        elif self.stage == Stage.AWAITING_SECRET:
            if len(self.buffer) == 99:
                if self.buffer[-1] != '\a' and self.buffer.find('\a\b') == -1:
                    raise ServerSyntaxError
            if len(self.buffer) >= 100:
                secret_len = self.buffer.find('\a\b')
                if secret_len == -1 or secret_len > 98:
                    raise ServerSyntaxError

        elif self.stage == Stage.MOVING:

        return True

    def _terminate(self):
        self.sock.close()

    def _parse_charging(self, message):
        if message == 'RECHARGING\a\b' and self.charging:
            if self.charging:
                raise ServerLogicError
            self.charging = True
            return True
        if message == 'FULL POWER\a\b':
            if not self.charging:
                raise ServerLogicError
            self.charging = False
            return True
        return False

    def _parse_move(self, message: str) -> np.ndarray:
        m = re.match(r'^OK (-?\d+) (-?\d+)$', message)
        if not m:
            raise ServerSyntaxError
        x, y = (int(i) for i in m.groups())
        return np.array((x, y))


def main():
    if len(sys.argv) < 3:
        print("Usage: <address> <port>", file=sys.stderr)
        return 1

    address = sys.argv[1]

    byte_regex = r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    IPv4_regex = r'\.'.join([byte_regex] * 4)

    m = re.match(IPv4_regex, address)
    if m is None:
        print("Usage: <address> <port>", file=sys.stderr)
        return 1

    for b in m.groups():
        if not 0 <= int(b) < 256:
            print("IP address out of range.", file=sys.stderr)
            return 1

    port = int(sys.argv[2])
    if port > 65535:
        print("Port number out of range.", file=sys.stderr)
        return 1

    server = Server(address, port)
    server.start()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    sys.exit(main())
