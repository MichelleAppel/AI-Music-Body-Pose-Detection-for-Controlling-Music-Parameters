from pythonosc import udp_client

class OSCSender:
    """
    A class for sending OSC messages over UDP.

    OSC (Open Sound Control) is a protocol for communication between computers, synthesizers, and other multimedia devices.
    """

    def __init__(self, ip_address, port):
        """
        Initialize the OSCSender with an IP address and port number.

        :param ip_address: The IP address to send OSC messages to.
        :param port: The port to send OSC messages to.
        """
        self.ip_address = ip_address
        self.port = port

        # Create a UDP client for sending OSC messages
        self.client = udp_client.SimpleUDPClient(self.ip_address, self.port)

    def send_message(self, address, data):
        """
        Send an OSC message with a given address and data.

        :param address: The OSC address for the message.
        :param data: The data to send with the message.
        """
        # Send the OSC message using the UDP client
        self.client.send_message(address, data)
