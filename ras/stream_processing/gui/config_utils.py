import json
import logging
import os
import platform
import re
import socket
from copy import deepcopy
from typing import Optional, Union, Callable
from zipfile import ZipFile

import cv2 as cv
import sounddevice as sd

from ..models.avatar.chrome_runner import get_web_extension_file
from ..utils import resolve_file_path


VOICE_FILE_EXTENSION = re.compile(r'.pt$', re.RegexFlag.IGNORECASE)

FEMALE_VOICES = {'tanja', 'wendy'}
MALE_VOICES = {'herbert', 'john'}

DEFAULT_AVATARS = {
    'Default Avatar (Female)':   'avatar_1_f.glb',
    'Default Avatar (Male)':     'avatar_2_m.glb',
    'Default Avatar 2 (Female)': 'avatar_3_f.glb',
    'Default Avatar 2 (Male)':   'avatar_4_m.glb',
}


def get_audio_devices(is_input: bool, logger: Optional[logging.Logger] = None) -> list[str]:
    """Retrieve the device names for audio-input or -output index"""
    devices = sd.query_devices()
    if logger:
        logger.debug('audio devices: %s', devices)
    result = []
    for idx, device in enumerate(devices):
        if is_input and device["max_input_channels"] > 0:
            result.append(device["name"])
        elif not is_input and device["max_output_channels"] > 0:
            result.append(device["name"])
    return result


def get_virtual_camera_backends() -> dict[str, Union[str, bool]]:
    result = {
        'DISABLED': False,
        'ENABLED <DEFAULT>': True,
    }
    # adapted from pyvirtualcam/camera.py:
    p = platform.system()
    if p == 'Windows':
        result['OBS Virtual Camera'] = 'obs'
        result['Unity Video Capture'] = 'unitycapture'
    elif p == 'Darwin':
        result['OBS Virtual Camera'] = 'obs'
    elif p == 'Linux':
        result['Loopback /dev/video<n>'] = 'v4l2loopback'
    return result


def is_port_valid(val: any) -> bool:
    """
    HELPER test configuration value for `video/converter/ws_port`

    :returns: `True` if it is an integer and a free port, otherwise `False`
    """
    if isinstance(val, int) and val > 0:
        return is_port_free(val)
    return False


def is_positive_number(val: any) -> bool:
    """
    HELPER test if value is a positive number

    :returns: `True` if it is a positive number (incl. zero), otherwise `False`
    """
    if val >= 0:
        return True
    return False


def is_positive_number_and_equal_to_config_path(val: any, config: Optional[dict], config_path: list[str]) -> bool:
    """
    HELPER test if value is a positive number AND (if `config` is not `None`) if it
           is equal to a value of another configuration-path

    E.g. for ensuring that `video/width` is same as `video/converter/width`.

    :param val: the value to test
    :param config: the configuration dictionary (if `None`, comparison to other value will be omitted)
    :param config_path: configuration path to other config-value that should be equal to `val`
    :returns: `True` if it is a positive number (incl. zero) and (if `config` is not `None`) equal to other
              configuration-path, otherwise `False`
    """
    if not is_positive_number(val):
        return False
    if config is None:
        return True
    other_val, _, _ = get_current_value_and_config_path_for(config, config_path)
    return other_val == val


def is_port_free(port: int) -> bool:
    """
    HELPER check if port is free

    adapted from:
    https://stackoverflow.com/a/43271125/4278324

    NOTE: referenced example using `Socket.connect_ex()` is slower when testing free ports

    :param port: the port number to check
    :returns: `True` if the `port` if free (i.e. can be opened to listen to)
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # NOTE: Socket.connect_ex() is actually slower than Socket.bind(), in cases where the port is free
        #       -> since we can assume, that the port is free in most cases, use `bind()` for quick testing
        try:
            s.bind(("0.0.0.0", port))
            result = True
        except Exception:
            result = False
        s.close()
        return result


def wrap_simple_validate(validate_func: Callable[[any], bool]) -> Callable[[any, Optional[dict]], bool]:
    """
    HELPER create wrapper function, to allow single-parameter `ConfigurationItem.is_valid_value(val)`
           (i.e. will always ignore `config`-parameter for `ConfigurationItem.is_valid_value(val, config)`)
    """
    def _func(val: any, _config: Optional[dict]):
        return validate_func(val)
    return _func


def get_current_value_and_config_path_for(config: dict, config_path: list[str]) -> tuple[any, str, dict]:
    """
    HELPER for a path in the config return:
     * the `current_value` at that path
     * the last containing dictionary `sub_config` that contains the value
     * the `field_name` within that `sub_config` that refers to the `current_value`

    i.e. something like
    sub_config ~> self.config[...config_path[:-1]]
    current_value = sub_config[config_path[-1:]

    :param config: the config dictionary
    :param config_path: the access path in the config
    :returns: a `tuple[<current_value>, <field_name>, <sub_config>]`
    """
    current_value = None
    field_name = None
    sub_config = config
    for curr_field in config_path:
        field_name = curr_field
        current_value = sub_config.get(field_name)
        if isinstance(current_value, dict):
            sub_config = current_value
    return current_value, field_name, sub_config


def get_voices_from(dir_path: str, logger: Optional[logging.Logger] = None) -> dict[str, str]:
    items: dict[str, str] = {}
    if os.path.exists(dir_path):
        for p in os.listdir(dir_path):
            file_path = os.path.join(dir_path, p)
            if os.path.isfile(file_path) and VOICE_FILE_EXTENSION.search(string=p):
                name = p[:-3].lower()
                if name in FEMALE_VOICES:
                    name = f'Female ({name.capitalize()})'
                elif name in MALE_VOICES:
                    name = f'Male ({name.capitalize()})'
                else:
                    name = name.capitalize()
                items[name] = file_path.replace('\\', '/')
    if not items:
        info_exists = ' (directory does not exist)' if not os.path.exists(dir_path) else (
            ' (path is not a directory)' if os.path.isdir(dir_path) else ''
        )
        msg = 'Failed to find any voices (*.pt files) at{}: {}'.format(info_exists, os.path.realpath(dir_path))
        # TODO raise exception that can be show to users in alert-box?
        if logger:
            logger.error(msg)
        else:
            print(msg)
    return items


def get_avatars_for_config(config: Optional[dict], logger: Optional[logging.Logger] = None) -> dict[str, str]:
    """
    HELPER inspect configuration and try to determine which avatar files are available.

    see also `get_avatars_from()` and `models.avatar.chrome_runner.get_web_extension_file()`

    :param config: the configuration (NOTE if omitted, the avatars are assumed to be in packed web-extension file)
    :param logger: OPTIONAL logger for debug/warn output
    :returns: a dictionary (labels -> avatar-files) for avatar-selection
              if no avatars could be found, the default avatars are returned, see `DEFAULT_AVATARS`
    """
    avatars: Optional[dict] = None
    if config:
        config_path = ['video', 'converter', 'use_chrome_extension']
        web_extension_config, _, _ = get_current_value_and_config_path_for(config, config_path)
        file_path = None
        if isinstance(web_extension_config, bool) and web_extension_config:
            file_path = get_web_extension_file()
        elif isinstance(web_extension_config, str):
            file_path = web_extension_config
            if not os.path.exists(file_path):
                file_path = resolve_file_path(file_path)
        elif logger:
            logger.error('Unknown configuration value for %s: expected bool or string, but got %s (%s)', '.'.join(config_path), type(web_extension_config), web_extension_config)

        if file_path:
            if os.path.exists(file_path):
                avatars = get_avatars_from(file_path, logger=logger)
            elif logger:
                logger.error('invalid configuration value for %s: file or directory does not exist at %s', '.'.join(config_path), file_path)
    else:
        # DEFAULT if config is omitted: try to use packed web-extension file
        file_path = get_web_extension_file()

    if not avatars:
        if logger:
            logger.warning('Could not detect any avatar models (*.glb), using default values instead. Did not find any at %s', file_path)
        avatars = deepcopy(DEFAULT_AVATARS)
    return avatars


def get_avatars_from(dir_or_zip_path: str, logger: Optional[logging.Logger] = None) -> dict[str, str]:
    """
    HELPER read available avatar files (3d models for avatars in `*.glb` format) from
           either packed web-extension file (ZIP) or unpacked web-extension directory

    Creates a dictionary with generated labels and the available avatars as values (file names, without paths)

    EXAMPLE:
    ```
       {
         'Avatar (Female)':      'avatar_1_f.glb',
         'Avatar (Male)':        'avatar_2_m.glb',
         'Avatar 2 (Female)':    'avatar_3_f.glb',
         'Avatar 2 (Male)':      'avatar_4_m.glb',
       }
    ```
    """
    avatars_list = []
    if os.path.isdir(dir_or_zip_path):
        for p in os.listdir(dir_or_zip_path):
            if os.path.isdir(os.path.join(dir_or_zip_path, p)):
                continue
            if p.lower().endswith('.glb'):
                avatars_list.append(p)
    else:
        with ZipFile(dir_or_zip_path) as zip_file:
            for n in zip_file.namelist():
                if n.lower().endswith('.glb'):
                    avatars_list.append(n)
    avatars_list.sort()

    female_count = 0
    male_count = 0
    unknown_count = 0
    result = {}

    # name pattern: "avatar_<number>_<gender>.glb"
    re_avatar = re.compile(r'avatar_(\d+)_(\w+)')

    for a in avatars_list:
        num = ''
        gender = ''
        match = re_avatar.search(a.lower())
        if match:
            # num = '' if match.group(1) == '1' else f'{match.group(1)} '
            # gender = 'Female' if match.group(2) == 'f' else 'Male'
            gender = match.group(2)
            if gender == 'm':
                male_count += 1
                gender = 'Male'
                num = '' if male_count == 1 else f'{male_count} '
            else:
                if gender == 'f':
                    female_count += 1
                    gender = 'Female'
                    num = '' if female_count == 1 else f'{female_count} '
                elif logger:
                    logger.warning('Unknown naming pattern for avatar files: expected "f" or "m" for gender, but got "%s" for file %s', gender, a)

        if not gender:
            unknown_count += 1
            gender = 'Unknown'
            num = '' if unknown_count == 1 else f'{unknown_count} '

        result[f'Avatar {num}({gender})'] = a
    return result


def validate_config_values(config: dict) -> list[str]:
    # NOTE use local import to avoid circular dependencies upon module initialization:
    from .config_items import CONFIG_ITEMS, IGNORE_CONFIG_ITEM_KEYS

    problems = []
    for key, item in CONFIG_ITEMS.items():
        if key in IGNORE_CONFIG_ITEM_KEYS or item.can_ignore_validation(config):
            continue
        curr_val, field, sub_config = get_current_value_and_config_path_for(config, item.config_path)
        if item.is_valid_value:
            if not item.is_valid_value(curr_val, config):
                problems.append('{}: {}'.format('.'.join(item.config_path), json.dumps(curr_val)))
        else:
            config_value_items = item.get(config)
            config_values = config_value_items if isinstance(config_value_items, list) else config_value_items.values()
            if curr_val not in config_values:
                problems.append('{}: {}'.format('.'.join(item.config_path), json.dumps(curr_val)))
    return problems


# adapted from
# https://stackoverflow.com/a/61768256/4278324
def return_camera_infos(logger: Optional[logging.Logger] = None, max_look_ahead: int = 2) -> list[dict]:
    """
    Get list of available cameras.

    NOTE: checks indices further down, even if one does not work (see `max_look_ahead`), since on some systems there seem to be "empty" indices
          see
          https://stackoverflow.com/a/61768256/4278324

    TODO create list with device names: not supported (yet?) by opencv2
    see possible solution for windows:
    https://github.com/yushulx/python-capture-device-list
    also possible bug/solution for this:
    https://github.com/yushulx/python-capture-device-list/issues/5

    possible solution for linux (could this work for macos?):
    https://github.com/opencv/opencv/issues/4269#issuecomment-1936742564
    utilizes library/package v4l-utils:
    v4l2-ctl --list-devices


    could try using ffmpeg, like
    ffmpeg -list_devices true -f dshow -i dummy
    see https://trac.ffmpeg.org/wiki/Capture/Webcam
    ... but does not seem to give information to which camera these correspond,
        using opencv2 which we are using for capturing

    **WARNING:** this function takes noticeable time to run
                 (since it does actually open cameras & captures for testing if camera is usable)

    :param logger: OPTIONAL logger for printing debug output
    :param max_look_ahead: OPTIONAL maximal look-ahead for checking camera indices: check at most these consecutive
                                    indices for non-working camera-indices, before stopping
                           (DEFAULT: `2`)
    :returns: list of camera infos `{'index': int, 'width': float, 'height': float, 'fps': float, 'backend': str}`
    """
    index = 0
    arr = []
    last_valid_idx = -1
    while True:
        cap = cv.VideoCapture(index)

        if logger and logger.isEnabledFor(logging.DEBUG):
            camera_info = (' | backend: "%s" | exception mode: %s' % (cap.getBackendName(), cap.getExceptionMode()) if cap.isOpened() else '')
            logger.debug('check camera %s -> opened: %s%s', index, cap.isOpened(), camera_info)

        if cap.isOpened() and cap.read()[0]:
            cap_info = {
                'index':    index,
                'backend':  cap.getBackendName(),

                # NOTE: these are the current default settings, when opening camera without specifying anything
                #         i.e. they do NOT necessarily reflect the camera's capabilities
                #         w.r.t. to min./max. resolution or FPS
                'width':    cap.get(cv.CAP_PROP_FRAME_WIDTH),
                'height':   cap.get(cv.CAP_PROP_FRAME_HEIGHT),
                'fps':      cap.get(cv.CAP_PROP_FPS),
            }
            arr.append(cap_info)
            cap.release()
            last_valid_idx = index
            is_current_valid = True
        else:
            is_current_valid = False
        index += 1

        # only continue testing, if last valid camera index is at most `max_look_ahead` steps back
        if not is_current_valid:
            if last_valid_idx == -1:
                # -> did not find any valid camera-indices yet!
                if index > max_look_ahead:
                    break
            else:
                # abort, if last valid camera-index is more than `max_look_ahead` steps back
                if index - last_valid_idx > max_look_ahead:
                    break
    return arr


def get_camera_device_items(logger: Optional[logging.Logger] = None) -> dict[str, int]:
    camera_infos = return_camera_infos(logger)
    items: dict[str, int] = {}
    for info in camera_infos:
        label = 'Camera {index} (width {width:.0f}, height {height:.0f}, FPS {fps})'.format(**info)
        items[label] = info['index']
    return items
