from pythonosc import udp_client

class OSCSender:
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port
        self.client = udp_client.SimpleUDPClient(self.ip_address, self.port)

    def send_message(self, address, data):
        self.client.send_message(address, data)
