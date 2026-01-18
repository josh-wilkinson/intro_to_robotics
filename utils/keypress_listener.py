import termios, fcntl, sys, os

# implementation found here:
# https://stackoverflow.com/questions/63605503/listen-for-a-specific-key-using-pynput-keylogger
class KeypressListener():
    def __init__(self) -> None:
        pass

    def get_keypress(self):
        chars = sys.stdin.readline()
        if len(chars) > 0:
            char = str(chars)[-1]
            return char
        else:
            return None

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.oldterm)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.oldflags)

    def __enter__(self):
        self.fd = sys.stdin.fileno()

        self.oldterm = termios.tcgetattr(self.fd)
        newattr = termios.tcgetattr(self.fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(self.fd, termios.TCSANOW, newattr)

        self.oldflags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.oldflags | os.O_NONBLOCK)

        print("\n") # we need this to be in the next line
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        pass


# if __name__ == '__main__':
#     keypress_listener = KeypressListener()
#     try:
#         while True:
#             try:
#                 char = keypress_listener.get_keypress()
#                 if char is not None:
#                     print("Got character", repr(char))
                    
#             except IOError: pass
#     finally:
#         keypress_listener.close()