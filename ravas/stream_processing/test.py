from torch.multiprocessing import Process, Queue
from threading import Event
import websockets
import logging
import time
from websocket_server import WebsocketServer


def server(input_queue: Queue):
    server = WebsocketServer(host="0.0.0.0", port=8887)
    server.set_fn_message_received(lambda client, server, message: print(message))

    client_available = Event()

    server.set_fn_new_client(lambda client, server: client_available.set())
    server.set_fn_client_left(lambda client, server: client_available.clear())
    server.run_forever(True)
    print("Server started")
    print("Waiting for clients to connect")
    client_available.wait()
    print("Client connected")
    while True:
        input_queue.get()


queue = Queue()
process = Process(target=server, args=(queue,))
process.start()
while True:
    # queue.put("Hello")
    time.sleep(10)
