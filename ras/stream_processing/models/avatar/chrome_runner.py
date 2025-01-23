import json
import logging
import os
import signal
import time
from enum import Enum
from typing import Optional, Union
from urllib.parse import quote

from selenium import webdriver
from torch.multiprocessing import Process, Event, Queue

from .avatar_resources import get_web_extension_file, get_web_extension_path
from ...dist_logging import worker_configurer


BASE_URL = 'http://localhost'
""" default URL when starting chrome driver with web URL (i.e. not with extension) """
PORT = 3000
""" default port when starting chrome driver with web URL (i.e. not with extension) """
WS_ADDR: Optional[str] = None  # 'http://127.0.0.1:8888'
""" default address for web-socket (i.e. connection to python process), if `None`, the react-app will use its default setting """
DEFAULT_EXTENSION_ID = 'mjioebaagpdpfjbicjponoajkbdphofk'
""" default extension ID for packed & signed chrome extension """

_chrome_version: str = ''
""" version (str) of the Chrome Broswer (will be detected, see `get_chrome_version()`) """


class StartExtensionType(Enum):
    NO_EXTENSION = 0
    PACKED_EXTENSION = 1
    UNPACKED_EXTENSION = 2


def get_chrome_version(logger: Optional[logging.Logger] = None) -> str:
    """
    HELPER detect Chrome Broswer version (as string)

    CAUTION: this can take considerable time (> 5 secs) for the first run
             (the result will be cached, so consecutive invocations will be faster)

    :returns: the Chrome Browser version as a string, or EMPTY string, if detection failed
    """
    global _chrome_version

    if not _chrome_version:
        # NOTE [russa] try detecting Chrome version, but only ONCE:
        #              since detection of Chrome capabilities takes a long time (> 5 secs), only do this once per run
        #              ... this will not take into account, if Chrome was updated in the meantime, but as a compromise
        #              the long-running detection is only done once
        if logger:
            logger.debug('Start detecting Chrome version...')
        # NOTE need to apply any fixed here already, to avoid any visibility problems
        #      IMPORTANT: must use `force`, otherwise it would this function circularly
        options = fix_chrome_options(webdriver.ChromeOptions(), True, force=True, logger=logger)
        driver = webdriver.Chrome(options=options)
        _chrome_version = driver.capabilities.get('browserVersion', '')
        driver.quit()
        if logger:
            logger.debug('Detected Chrome version: %s', _chrome_version)
    return _chrome_version


def create_options(start_extension: Optional[StartExtensionType] = None, extension_path: Optional[str] = None, run_headless: bool = True, debug_port: Optional[int] = None, logger: Optional[logging.Logger] = None) -> webdriver.ChromeOptions:
    options = webdriver.ChromeOptions()

    if start_extension:
        if start_extension == StartExtensionType.PACKED_EXTENSION:
            ext_path = get_web_extension_file() if not extension_path else extension_path
            if not os.path.exists(ext_path):
                msg = 'Invalid path for packed web extension (file does not exist): "%s"'
                if logger:
                    logger.error(msg, ext_path)
                else:
                    print('ERROR ' + msg % ext_path, flush=True)
            options.add_extension(ext_path)
        elif start_extension == StartExtensionType.UNPACKED_EXTENSION:
            ext_path = get_web_extension_path() if not extension_path else extension_path
            if not os.path.exists(ext_path):
                msg = 'Invalid path for unpacked web extension (directory does not exist): "%s"'
                if logger:
                    logger.error(msg, ext_path)
                else:
                    print('ERROR ' + msg % ext_path, flush=True)
            options.add_argument(f'load-extension="{ext_path}"')

    if run_headless:
        options.add_argument("--headless=new")

    if isinstance(debug_port, int):
        options.add_argument('--remote-debugging-port={}'.format(debug_port))

    # apply ony fixes, if necessary:
    fix_chrome_options(options, run_headless, logger=logger)

    # options.add_argument("--virtual-time-budget=30000")  # EXPERIMENTAL try to fast-forward internal browser clock [russa: does not seem to work for our purposes]

    # EXPERIMENTAL: configure browser to allow capturing video/audio from webcam/microphone
    # options.add_experimental_option("prefs", {
    #     "profile.default_content_setting_values.media_stream_mic": 1,     # 1:allow, 2:block
    #     "profile.default_content_setting_values.media_stream_camera": 1,  # 1:allow, 2:block
    #     "profile.default_content_setting_values.geolocation": 2,          # 1:allow, 2:block
    #     "profile.default_content_setting_values.notifications": 2         # 1:allow, 2:block
    # })
    return options


def fix_chrome_options(options: webdriver.ChromeOptions, is_headless: bool, force: bool = False, logger: Optional[logging.Logger] = None) -> webdriver.ChromeOptions:
    """
    HELPER fix problems for Chrome by apply additional arguments, if necessary

    :param options: the options to fix (INOUT parameter)
    :param is_headless: use `True` if headless-mode is or will be enabled (in the `options`)
    :param force: set to `True` to prevent detection (if patches are necessary) and instead force applying all patches
    :param logger: OPTIONAl logger for debug output
    :returns: the `options` (with applied fixes, if necessary)
    """
    # do NOT call get_chrome_version() if force is enabled (otherwise it will cause circular invocations):
    chrome_version = get_chrome_version(logger) if not force else ''
    if is_headless or force:
        if not chrome_version or chrome_version.startswith('129.') or force:
            # HACK for bug in Chrome 129.x on Windows, showing a blank window in new-headless mode:
            #      WORKAROUND move window off-screen
            #      see
            #      https://stackoverflow.com/a/78999088/4278324
            if logger:
                logger.debug('Applying PATCH for Chrome version is 129.x: fix BUG that shows blank window')
            options.add_argument("--window-position=-2400,-2400")
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


def finish_browser(driver: webdriver.Chrome, logger: logging.Logger):

    # before quitting: print some log information
    try:
        info = driver.execute_script('return window["info_init"]')
        logger.info('web info %s (ms): %s', 'INIT', info)

        # info = driver.execute_script('return window["info_detect"]')
        # logger.info('web info %s (ms / frames): %s', 'DETECT', info)

        info = driver.execute_script('return window["info_render"]')
        logger.info('web info %s (ms / frames): %s', 'RENDER', info)

    except Exception as exc:
        logger.warning('failed to get info from web window, due to error ', exc_info=exc)

    # do quit browser:
    driver.quit()


def start_browser(
        ws_addr: Optional[str] = WS_ADDR,
        stop_signal: Queue = Queue(),
        port: int = PORT,
        base_url: str = BASE_URL,
        web_extension: Union[bool, str] = False,
        run_headless: bool = True,
        avatar_uri: Optional[str] = None,
        hide_avatar_selection: Optional[bool] = None,
        log_queue: Optional[Queue] = None,
        log_level: Optional[str] = None,
):
    """
    Start a (Google) Chrome browser instance for rendering avatar images.

    :param ws_addr: the web-socket address for receiving the avatar-pose-data & sending the rendered avatar images
    :param stop_signal: signal for stopping the chrome browser / process: sending the value `None` will stop the browser
    :param port: the port for opening the avatar-rendering web app with `<base_url>:<port>`
                 (only used, if _not_ started with web-extension)
    :param base_url: the URL for opening the avatar-rendering web app with `<base_url>:<port>`
                     (only used, if _not_ started with web-extension)
    :param web_extension: either `bool` or a `str`ing:
                         * if `False` will __not__ start Chrome with the web extension for avatar-rendering, but instead
                           start as a normal website with `"<base_url>:<port>"`
                           (i.e. that URL should be accessible from this machine)
                         * if `True` will start Chrome with the _packed_ web extension
                           (the `*.crx` file is assumed at the default location in the parallel subproject
                           `rpm/dist/chrome-extension.crx`, see `get_web_extension_file()`)
                         * if `str` and if it is a valid path:
                           * if it is a path to a _directory_, it will be loaded as _unpacked_ web-extension
                           * if it is a path to a _file_, it will be loaded as a _packed_ web-extension
                         * if `str` but is _no_ valid path:
                           the web-extension will be loaded as _packed_ web-extension, and the `str`ing is used as
                           custom extension ID, i.e. chrome browser will be started with the URL
                           `chrome-extension://<extension ID>/index.html` for opening the the extension
                           (the `*.crx` file is assumed at the default location in the parallel subproject
                           `rpm/dist/chrome-extension.crx`, see `get_web_extension_file()`)
    :param run_headless: if `True`, run Chrome in `headless` mode, i.e. "invisible" without showing its window
    :param avatar_uri: OPTIONAL the URI for loading the avatar (`*.glb` file)
    :param hide_avatar_selection: OPTIONAL if `True` hide widgets for selecting an avatar
                                    (without these widgets, you need to use the query-parameters for setting an avatar).
                                  If `None` (or omitted), will hide the widgets if run in `headless` mode (and show
                                  them if _not_ run in `headless` mode).
    :param log_queue: OPTIONAL a `Queue` for multiprocessing logging via queue
    :param log_level: OPTIONAL the log-level for multiprocessing logging
    """
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

        options = create_options(start_extension=start_extension, extension_path=extension_path, run_headless=run_headless, logger=logger)
        driver = webdriver.Chrome(options=options)

        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Chrome driver capabilities:\n%s', json.dumps(driver.capabilities))

        if start_extension == StartExtensionType.NO_EXTENSION:
            if port >= 0:
                target_url = '{}:{}'.format(base_url, port)
            else:
                target_url = base_url
        else:
            target_url = 'chrome-extension://{}/index.html'.format(extension_id)

        query_params = []
        if ws_addr:
            query_params.append('ws=' + quote(ws_addr))
        if not run_headless:
            query_params.append('show-fps=' + quote('true'))
        if avatar_uri:
            query_params.append('avatar=' + quote(avatar_uri))
        if hide_avatar_selection is True or (hide_avatar_selection is None and run_headless):
            # NOTE default value for "show-selection" is TRUE, so omitting it is the same as setting it to true
            query_params.append('hide-selection=' + quote('true'))
        if len(query_params) > 0:
            target_url += '?' + '&'.join(query_params)

        logger.info('starting chrome driver for URL: %s', target_url)
        driver.get(target_url)

        while True:
            try:
                # stop signal: receiving the value None
                if stop_signal.get(timeout=1) is None:
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
                logger.info('INTERRUPTED, stopped chrome driver service!')
            else:
                try:
                    driver.service.assert_process_still_running()
                    finish_browser(driver, logger)
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
