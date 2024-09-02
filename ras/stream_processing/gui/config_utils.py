import logging
import os
import re
from typing import Optional

import cv2 as cv
import sounddevice as sd


VOICE_FILE_EXTENSION = re.compile(r'.pt$', re.RegexFlag.IGNORECASE)

FEMALE_VOICES = {'tanja', 'wendy'}
MALE_VOICES = {'herbert', 'john'}


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



"""
adapted from
https://stackoverflow.com/a/61768256/4278324

NOTE: checks indices further down, even one does not work, since on systems there seem to be "empty" indices, see
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
"""
def return_camera_indices(logger: Optional[logging.Logger] = None) -> list[str]:
    # checks the first 10 indexes.
    # WARNING: this function takes noticeable time to run
    #          (since it does actually open cameras & captures for testing if camera is usable)
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv.VideoCapture(index)

        if logger and logger.isEnabledFor(logging.DEBUG):
            camera_info = (' | backend: "%s" | exception mode: %s' % (cap.getBackendName(), cap.getExceptionMode()) if cap.isOpened() else '')
            logger.debug('check camera %s -> opened: %s%s', index, cap.isOpened(), camera_info)

        if cap.isOpened() and cap.read()[0]:
            arr.append(str(index))
            cap.release()
        index += 1
        i -= 1
    return arr
