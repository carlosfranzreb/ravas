import http.server
import logging
import os
import socketserver
from argparse import ArgumentParser
from typing import Optional

from torch.multiprocessing import Queue

from ...dist_logging import worker_configurer


PORT = 3000
DIRECTORY = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', "..", "rpm", "build"))


# adapted from
# https://stackoverflow.com/a/52531444/4278324
def handler_for_dir(directory: str):
    def _init(self, *args, **kwargs):
        return http.server.SimpleHTTPRequestHandler.__init__(self, *args, directory=self.directory, **kwargs)
    return type(f'HandlerForDirectory<{directory}>',
                (http.server.SimpleHTTPRequestHandler,),
                {'__init__': _init, 'directory': directory})


def start_server(port: int = PORT, directory: str = DIRECTORY, log_queue: Optional[Queue] = None, log_level: Optional[str] = None):

    if log_queue:
        worker_configurer(log_queue, log_level if log_level else 'INFO')
    logger = logging.getLogger("web_server")

    with socketserver.TCPServer(("", port), handler_for_dir(directory)) as httpd:
        logger.info("starting to serve at port %s, contents of directory %s", port, directory)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("caught KeyboardInterrupt, stopping server...")
            httpd.shutdown()
            logger.info("server stopped.")


if __name__ == "__main__":
    parser = ArgumentParser('Start static file server for React-App (for Avatar Rendering)')
    parser.add_argument('--port', '-p', type=int, default=PORT)
    parser.add_argument('--target-dir', '-t', type=str, default=DIRECTORY)
    p_args = parser.parse_args()
    port_num = p_args.port
    target = p_args.target_dir
    logger = logging.getLogger('http-server')
    try:
        start_server(port=port_num, directory=target)
    except Exception as exc:
        logger.error('Encountered error when starting server', exc_info=exc, stack_info=True)
