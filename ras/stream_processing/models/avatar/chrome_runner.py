import logging
import os
import signal
import time
from enum import Enum
from typing import Optional, Union

from selenium import webdriver
from torch.multiprocessing import Process, Event, Queue

from ...dist_logging import worker_configurer


BASE_URL = 'http://localhost'
""" default URL when starting chrome driver with web URL (i.e. not with extension) """
PORT = 3000
""" default port when starting chrome driver with web URL (i.e. not with extension) """
WS_ADDR: Optional[str] = None  # 'http://107.0.0.1:8888'
""" default address for web-socket (i.e. connection to python process), if `None`, the react-app will use its default setting """
DEFAULT_EXTENSION_ID = 'mjioebaagpdpfjbicjponoajkbdphofk'
""" default extension ID for packed & signed chrome extension """

PROJECT_BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
""" root directory of project """


class StartExtensionType(Enum):
    NO_EXTENSION = 0
    PACKED_EXTENSION = 1
    UNPACKED_EXTENSION = 2


def get_web_extension_path() -> str:
    return os.path.join(PROJECT_BASE_DIR, 'rpm', 'dist', 'chrome-extension');


def get_web_extension_file() -> str:
    return os.path.join(PROJECT_BASE_DIR, 'rpm', 'dist', 'chrome-extension.crx');


def create_options(start_extension: Optional[StartExtensionType] = None, extension_path: Optional[str] = None, run_headless: bool = True, debug_port: Optional[int] = None) -> webdriver.ChromeOptions:
    options = webdriver.ChromeOptions()

    if start_extension:
        if start_extension == StartExtensionType.PACKED_EXTENSION:
            options.add_extension(get_web_extension_file() if not extension_path else extension_path)
        elif start_extension == StartExtensionType.UNPACKED_EXTENSION:
            options.add_argument(f"load-extension={get_web_extension_path() if not extension_path else extension_path}")

    if run_headless:
        options.add_argument("--headless=new")

    if isinstance(debug_port, int):
        options.add_argument('--remote-debugging-port={}'.format(debug_port))

    # options.add_argument("--virtual-time-budget=30000")  # EXPERIMENTAL try to fast-forward internal browser clock [russa: does not seem to work for our purposes]

    # EXPERIMENTAL: configure browser to allow capturing video/audio from webcam/microphone
    # options.add_experimental_option("prefs", {
    #     "profile.default_content_setting_values.media_stream_mic": 1,     # 1:allow, 2:block
    #     "profile.default_content_setting_values.media_stream_camera": 1,  # 1:allow, 2:block
    #     "profile.default_content_setting_values.geolocation": 2,          # 1:allow, 2:block
    #     "profile.default_content_setting_values.notifications": 2         # 1:allow, 2:block
    # })
    return options


# EXPERIMENTAL select a camera in chrome instance, see
# https://stackoverflow.com/a/69586993
def select_camera(driver: webdriver.Chrome, selected_camera: str):
    config_camera_url = "chrome://settings/content/camera"
    driver.get(config_camera_url)
    time.sleep(3)  # Wait until selector appears
    selector = driver.execute_script(
        "return document.querySelector('settings-ui')"
        ".shadowRoot.querySelector('settings-main')"
        ".shadowRoot.querySelector('settings-basic-page')"
        ".shadowRoot.querySelector('settings-section > settings-privacy-page')"
        ".shadowRoot.querySelector('settings-animated-pages > settings-subpage > media-picker')"
        ".shadowRoot.querySelector('#picker > #mediaPicker')"
        ".value = '{camera}'".format(camera=selected_camera)  # Change for your default camera
    )  # FIXME should check the option's textContent, since value might be some internal system ID
    #          (and not the visible text)

    # TODO implement alternative selection by index -> get option-list
    # ... .querySelectorAll('option')[index].value


def finish_browser(driver: webdriver.Chrome):
    # do quit browser:
    driver.quit()


def start_browser(ws_addr: Optional[str] = WS_ADDR, stop_signal: Queue = Queue(), port: int = PORT, base_url: str = BASE_URL, web_extension: Union[bool, str] = False, log_queue: Optional[Queue] = None, log_level: Optional[str] = None):

    if log_queue:
        worker_configurer(log_queue, log_level if log_level else 'INFO')
    logger = logging.getLogger("chrome_driver_renderer")

    driver = None
    was_interrupted = False
    try:

        start_extension = StartExtensionType.NO_EXTENSION
        extension_path = None
        extension_id = DEFAULT_EXTENSION_ID
        if isinstance(web_extension, bool):
            if web_extension:
                start_extension = StartExtensionType.PACKED_EXTENSION
        elif isinstance(web_extension, str):
            if os.path.exists(web_extension):
                start_extension = StartExtensionType.PACKED_EXTENSION if os.path.isfile(web_extension) else StartExtensionType.UNPACKED_EXTENSION
                extension_path = web_extension
            else:
                start_extension = StartExtensionType.PACKED_EXTENSION
                extension_id = web_extension

        options = create_options(start_extension=start_extension, extension_path=extension_path)
        driver = webdriver.Chrome(options=options)

        if start_extension == StartExtensionType.NO_EXTENSION:
            if port >= 0:
                target_url = '{}:{}'.format(base_url, port)
            else:
                target_url = base_url
        else:
            target_url = 'chrome-extension://{}/index.html'.format(extension_id)

        if ws_addr:
            target_url += '?ws=' + ws_addr

        logger.info('starting chrome driver for URL %s... ', target_url)
        driver.get(target_url)

        while True:
            try:
                if stop_signal.get(timeout=1):
                    break
            except:
                pass

        logger.debug('finished main loop... ')

    except KeyboardInterrupt:
        logger.info('caught KeyboardInterrupt')
        was_interrupted = True
    except Exception as exc:
        logger.info('caught Exception', exc_info=exc, stack_info=True)
        was_interrupted = True
    finally:
        if driver:
            logger.info('stopping chrome driver...')
            if was_interrupted:
                driver.service.stop()
                logger.info('INTERRUPTED, stopped chrome driver service, forcing exit!')
                import sys
                sys.exit(1)
            else:
                try:
                    driver.service.assert_process_still_running()
                    finish_browser(driver)
                    logger.info('stopped chrome driver.')
                except Exception as exc:
                    logger.info('encountered ERROR because chrome driver already stopped: ', exc_info=exc, stack_info=True)
        else:
            logger.info('did not yet start chrome driver, nothing to stop.')


# ############################### FOR running as script ########################


_cancel_count = 0


def start_main():
    import sys

    e = Event()

    def quit_browser(sig_num, frame):
        global _cancel_count
        _cancel_count += 1
        print('  start_main: DID receive signal "{}" (count: {})... '.format(sig_num, _cancel_count), flush=True)
        if not e.is_set():
            e.set()
        if _cancel_count > 1:
            print('  start_main: forcing exit ({})... '.format(_cancel_count), flush=True)
            raise KeyboardInterrupt('Forced Exit!')

    signal.signal(signal.SIGINT, quit_browser)
    signal.signal(signal.SIGTERM, quit_browser)

    p = Process(target=start_browser, args=(e, ), name='avatar_renderer')
    p.start()
    try:
        while not e.is_set():
            time.sleep(5)

    except Exception as err:
        print('start_main: Stopping... ')  # , err)
        if not e.is_set():
            e.set()
    finally:
        print('start_main: Waiting... ')
        if p.is_alive():
            p.join(20)
        if p.is_alive():
            print('start_main: Force stopping...')
            p.terminate()

        sys.stdout.flush()
        print('start_main: Stopped!')
        print('start_main: Exit Code: ', p.exitcode)


async def start_main_async():
    start_main()


if __name__ == "__main__":
    # driver = None
    # try:
    #     options = webdriver.ChromeOptions()
    #     options.add_argument("--headless=new")
    #     driver = webdriver.Chrome(options=options)
    #     driver.get('http://localhost:3000')
    #     while True:
    #         time.sleep(5)
    # except Exception as exc:
    #     print('Exception', exc)
    #     if driver:
    #         driver.quit()

    import asyncio
    asyncio.run(start_main_async())
    print('__main__: exit main!')
