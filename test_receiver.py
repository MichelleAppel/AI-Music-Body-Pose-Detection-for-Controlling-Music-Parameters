from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def print_message(address, *args):
    print(f"{address}: {args}")

if __name__ == '__main__':
    dispatcher = Dispatcher()
    dispatcher.map("*", print_message)
    server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)
    print("Listening for OSC messages on port 9000...")
    server.serve_forever()
