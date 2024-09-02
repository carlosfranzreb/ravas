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
