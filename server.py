#! venv/bin/python3
from __future__ import annotations
import re
import sys
import socket
import logging
import numpy as np
from enum import Enum
from threading import Thread


class ServerMessages(Enum):
    SERVER_CONFIRMATION = '{}'
    SERVER_MOVE = '102 MOVE'
    SERVER_TURN_LEFT = '103 TURN LEFT'
    SERVER_TURN_RIGHT = '104 TURN RIGHT'
    SERVER_PICK_UP = '105 GET MESSAGE'
    SERVER_LOGOUT = '106 LOGOUT'
    SERVER_KEY_REQUEST = '107 KEY REQUEST'
    SERVER_OK = '200 OK'
    SERVER_LOGIN_FAILED = '300 LOGIN FAILED'
    SERVER_SYNTAX_ERROR = '301 SYNTAX ERROR'
    SERVER_LOGIC_ERROR = '302 LOGIC ERROR'
    SERVER_KEY_OUT_OF_RANGE_ERROR = '303 KEY OUT OF RANGE'


class ServerLoginFailedError(Exception):
    ...


class ServerSyntaxError(Exception):
    ...


class ServerLogicError(Exception):
    ...


class ServerKeyOutOfRangeError(Exception):
    ...


class Server:
    def __init__(self, address: str, port: int):
        logging.info(f'{address=}, {port=}')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((address, port))
        self.sock = sock
        self.threads = []

    def start(self):
        try:
            self.sock.listen(1024)
            logging.info('Server is listening.')

            while True:
                conn, remote = self.sock.accept()
                logging.info(f'New connection: {remote}')

                t = Thread(target=Communication(conn, remote).start)
                t.start()
                self.threads.append(t)
        except KeyboardInterrupt:
            logging.info('Closing server...')
            for thread in self.threads:
                thread.join()


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


class Robot:
    def __init__(self):
        self.rotation = None
        self.position = None
        self.stuck = False
        self.move = None

    def add_position(self, position: np.ndarray):
        if self.position is None:
            self.position = position
            return self._at_start

        if self.rotation is None:
            dp = position - self.position
            self.position = position
            if np.all(dp == (0, 0)):
                self.stuck = self.move == ServerMessages.SERVER_MOVE
                return self._at_start

            self.stuck = False
            self.rotation = dp
            return self._at_start

        if np.all(position == self.position) and self.move == ServerMessages.SERVER_MOVE:
            self.stuck = True
            return self._at_start

        if self.stuck and self.move in (ServerMessages.SERVER_TURN_RIGHT, ServerMessages.SERVER_TURN_LEFT):
            return self._at_start

        self.stuck = False
        self.position = position
        return self._at_start

    @property
    def _at_start(self):
        if self.position is None:
            return False
        return np.all(self.position == (0, 0))

    def get_move(self):
        if self.position is None:
            self.move = ServerMessages.SERVER_TURN_RIGHT
            return self.move

        if self.rotation is None:
            if self.stuck:
                self.move = ServerMessages.SERVER_TURN_LEFT
                return self.move
            self.move = ServerMessages.SERVER_MOVE
            return self.move

        if self.stuck or dist_from_centre(self.position + self.rotation) >= dist_from_centre(self.position):
            if self.stuck and self.move in (ServerMessages.SERVER_TURN_RIGHT, ServerMessages.SERVER_TURN_LEFT):
                self.move = ServerMessages.SERVER_MOVE
                return self.move
            return self._get_next_rotation()

        self.move = ServerMessages.SERVER_MOVE
        return self.move

    def _get_next_rotation(self):
        next_rot = Rotation.turn_left(self.rotation)
        if dist_from_centre(self.position + next_rot) < dist_from_centre(self.position):
            self.rotation = next_rot
            self.move = ServerMessages.SERVER_TURN_LEFT
            return self.move
        else:
            self.rotation = Rotation.turn_right(self.rotation)
            self.move = ServerMessages.SERVER_TURN_RIGHT
            return self.move


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
        self.sock.settimeout(1)
        while True:
            try:
                data = self.sock.recv(1024)
            except TimeoutError:
                self._close()
                break

            if not data:
                self._close()
                break

            logging.debug(f'Message from {self.addr}: {data}')
            stream = data.decode()

            if not self._parse_stream(stream):
                return

    def _parse_stream(self, stream: str) -> bool:
        """ Return False if connection should be terminated. """
        self.buffer += stream
        try:
            while True:
                if not self._validate_buffer():
                    return False
                idx = self.buffer.find('\a\b')
                if idx == -1:
                    return True

                message = self.buffer[:idx]
                self.buffer = self.buffer[idx+2:]
                if not self._parse_message(message):
                    return False

        except ServerLoginFailedError:
            self._send(ServerMessages.SERVER_LOGIN_FAILED)

        except ServerSyntaxError:
            self._send(ServerMessages.SERVER_SYNTAX_ERROR)

        except ServerLogicError:
            self._send(ServerMessages.SERVER_LOGIC_ERROR)

        except ServerKeyOutOfRangeError:
            self._send(ServerMessages.SERVER_KEY_OUT_OF_RANGE_ERROR)

        return False

    def _parse_message(self, message: str) -> bool:
        """ Return False if connection should be terminated. """
        logging.debug(f'Processing {message}')
        if self._parse_charging(message):
            return True

        if self.stage == Stage.INITIAL:
            return self._parse_name(message)
        if self.stage == Stage.AWAITING_KEY_ID:
            return self._parse_key_id(message)
        if self.stage == Stage.AWAITING_CLIENT_HASH:
            return self._parse_client_hash(message)
        if self.stage == Stage.MOVING:
            position = self._parse_move(message)
            if self.robot.add_position(position):
                self.stage = Stage.AWAITING_SECRET
                return self._send(ServerMessages.SERVER_PICK_UP)

            move = self.robot.get_move()
            return self._send(move)

        if self.stage == Stage.AWAITING_SECRET:
            return self._send(ServerMessages.SERVER_LOGOUT)

        raise ValueError('Unknown stage.')

    def _parse_name(self, message: str) -> bool:
        self.remote_name = message
        self.name_hash = sum(ord(c) for c in message) * 1000 % Communication.MODULO
        self.stage = Stage.AWAITING_KEY_ID
        return self._send(ServerMessages.SERVER_KEY_REQUEST)

    def _parse_key_id(self, message: str) -> bool:
        try:
            self.key_id = int(message)
        except ValueError:
            logging.debug("Wrong key id.")
            raise ServerSyntaxError
        if self.key_id > 4:
            raise ServerKeyOutOfRangeError
        self.key_pair = self._key_pairs[self.key_id]
        server_hash = (self.key_pair[0] + self.name_hash) % Communication.MODULO
        self.stage = Stage.AWAITING_CLIENT_HASH
        return self._send(ServerMessages.SERVER_CONFIRMATION, server_hash)

    def _parse_client_hash(self, message: str) -> bool:
        m = re.match(r'^(\d{1,5})$', message)
        if not m:
            logging.debug("Wrong client hash.")
            raise ServerSyntaxError

        message = int(m.group())
        client_hash = (self.key_pair[1] + self.name_hash) % Communication.MODULO
        if client_hash != message:
            raise ServerLoginFailedError

        self._send(ServerMessages.SERVER_OK)

        self.stage = Stage.MOVING
        move = self.robot.get_move()
        return self._send(move)

    def _parse_charging(self, message):
        if self.charging and message != 'FULL POWER':
            raise ServerLogicError

        if message == 'RECHARGING':
            if self.charging:
                raise ServerLogicError
            logging.info('Recharging')
            self.charging = True
            self.sock.settimeout(5)
            return True
        if message == 'FULL POWER':
            if not self.charging:
                raise ServerLogicError
            logging.info('Full power')
            self.charging = False
            self.sock.settimeout(1)
            return True
        return False

    @classmethod
    def _parse_move(cls, message: str) -> np.ndarray:
        m = re.match(r'^OK (-?\d+) (-?\d+)$', message)
        if not m:
            logging.debug("Wrong move.")
            raise ServerSyntaxError
        x, y = (int(i) for i in m.groups())
        return np.array((x, y))

    def _send(self, message: ServerMessages, *args):
        close = message in (ServerMessages.SERVER_LOGIN_FAILED,
                            ServerMessages.SERVER_SYNTAX_ERROR,
                            ServerMessages.SERVER_LOGIC_ERROR,
                            ServerMessages.SERVER_KEY_OUT_OF_RANGE_ERROR,
                            ServerMessages.SERVER_LOGOUT)

        message = message.value.format(*args) + '\a\b'
        logging.debug(f'Sending: {message}')
        self.sock.send(message.encode())
        if close:
            self._close()

        return not close

    def _close(self):
        logging.info('Closing connection...')
        self.sock.close()

    def _validate_buffer(self):
        if not self.buffer:
            return True
        recharging = 'RECHARGING\a\b'
        if recharging.startswith(self.buffer[:len(recharging)]):
            return True

        full_power = 'FULL POWER\a\b'
        if full_power.startswith(self.buffer[:len(full_power)]):
            return True

        if self.stage == Stage.INITIAL:
            if len(self.buffer) == 19:
                if self.buffer[-1] != '\a' and self.buffer.find('\a\b') == -1:
                    logging.debug("Invalid buffer.")
                    raise ServerSyntaxError
            if len(self.buffer) >= 20:
                text_len = self.buffer.find('\a\b')
                if text_len == -1 or text_len > 18:
                    logging.debug("Invalid buffer.")
                    raise ServerSyntaxError

        elif self.stage == Stage.AWAITING_SECRET:
            if len(self.buffer) == 99:
                if self.buffer[-1] != '\a' and self.buffer.find('\a\b') == -1:
                    logging.debug("Invalid buffer.")
                    raise ServerSyntaxError
            if len(self.buffer) >= 100:
                secret_len = self.buffer.find('\a\b')
                if secret_len == -1 or secret_len > 98:
                    logging.debug("Invalid buffer.")
                    raise ServerSyntaxError

        elif self.stage == Stage.MOVING:
            matchers = (r'(^O$)',
                        r'(^OK$)',
                        r'(^OK -?$)',
                        r'(^OK -?\d{1,6}$)',
                        r'(^OK -?\d{1,6} -?$)',
                        r'(^OK -?\d{1,6} -?\d{1,6}$)',
                        r'(^OK -?\d{1,6} -?\d{1,6}\x07$)',
                        r'(^OK -?\d{1,6} -?\d{1,6}\x07\x08$).*')
            matches = [re.match(matcher, self.buffer) for matcher in matchers]
            if all(match is None for match in matches):
                logging.debug("Invalid buffer.")
                raise ServerSyntaxError

            match, = (match for match in matches if match is not None)
            if len(match.groups()[0]) > 12:
                logging.debug("Invalid buffer.")
                raise ServerSyntaxError

        return True


def main():
    if len(sys.argv) < 3:
        print("Usage: <address> <port>", file=sys.stderr)
        return 1

    address = sys.argv[1]

    byte_regex = r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    IPv4_regex = '^' + r'\.'.join([byte_regex] * 4) + '$'

    m = re.match(IPv4_regex, address)
    if m is None:
        print("Wrong IP address", file=sys.stderr)
        return 1

    port = int(sys.argv[2])
    if port > 65535:
        print("Port number out of range.", file=sys.stderr)
        return 1

    server = Server(address, port)
    server.start()
    return 0


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main())
