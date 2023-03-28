# This script sets up an OSC (Open Sound Control) server that listens for incoming messages on port 9000.
# The server uses the python-osc package to receive and process OSC messages.
# When a message is received, it is passed to the print_message() function, which simply prints the message to the console.

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# Define a function to print the address and arguments of the received OSC message
def print_message(address, *args):
    print(f"{address}: {args}")

if __name__ == '__main__':
    # Create a new dispatcher to handle incoming OSC messages
    dispatcher = Dispatcher()

    # Map all incoming OSC addresses to the print_message() function
    dispatcher.map("*", print_message)

    # Create a new OSC server that listens on all available network interfaces on port 9000
    server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)

    # Print a message to the console to indicate that the server is running and listening for messages
    print("Listening for OSC messages on port 9000...")

    # Start the OSC server and block the main thread to keep the server running
    server.serve_forever()
