import pytest
import sounddevice as sd
from ravas.stream_processing.gui.config_utils import get_audio_devices, return_camera_infos

'''
Test for get Audio Devices(input/output) and return_camera_infos
'''

def test_get_input_device():
    '''
        Test for get_audio_devices if is_input is True
        -Checks that the returned devices are a subset of the real input devices
        -Checks that there are no duplicate device names
        -Checks that the devices are filtered for input devices and hostapi
    '''
    real_input_device = sd.query_devices()
    real_default_api = sd.default.hostapi

    devices = get_audio_devices(is_input=True)

    valid_indices = {
        i for i, d in enumerate(real_input_device)
        if d["hostapi"] == real_default_api and d["max_input_channels"] > 0
    }

    assert set(devices.values()).issubset(valid_indices)

    names = [label.split(". ", 1)[1].rsplit(" [ID", 1)[0] for label in devices.keys()]

    # check for duplicates
    duplicates = {name for name in names if names.count(name) > 1}

    assert not duplicates, f"Duplicate device names found: {duplicates}"

    # check for filtering for input devices and hostapi
    for idx in devices.values():
        device = real_input_device[idx]
        assert device["hostapi"] == real_default_api
        assert device["max_input_channels"] > 0

def test_get_output_device():
    """
    Test for get_audio_devices if is_input is False
    - Checks that the returned devices are a subset of the real output devices
    - Checks that there are no duplicate device names
    - Checks that the devices are filtered for output devices and hostapi
    """
    real_output_devices = sd.query_devices()
    default_api = sd.default.hostapi

    devices = get_audio_devices(is_input=False)

    valid_indices = {
        i for i, d in enumerate(real_output_devices)
        if d["hostapi"] == default_api and d["max_output_channels"] > 0
    }

    # Check returned indices are a subset of valid ones
    assert set(devices.values()).issubset(valid_indices), "Returned indices include invalid output devices"

    # Extract device names from labels and check for duplicates
    names = [label.split(". ", 1)[1].rsplit(" [ID", 1)[0] for label in devices.keys()]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"Duplicate device names found: {duplicates}"

    # Check that hostapi is default and max_output_channels > 0
    for idx in devices.values():
        device = real_output_devices[idx]
        assert device["hostapi"] == default_api, f"Device {idx} does not match default hostapi"
        assert device["max_output_channels"] > 0, f"Device {idx} has no output channels"

def test_return_camera_infos_structure():
    '''
    Test for return_camera_infos
    - Checks that the returned value is a list of dictionaries with required keys
    - Checks that the index is a non-negative integer and that width, height, fps are numbers
    '''
    cameras = return_camera_infos()
    
    # Should return a list
    assert isinstance(cameras, list)

    for cam in cameras:
        # Each item should be a dict
        assert isinstance(cam, dict)

        # Must have required keys
        for key in ("index", "backend", "width", "height", "fps"):
            assert key in cam

        # index should be a non-negative integer
        assert isinstance(cam["index"], int)
        assert cam["index"] >= 0

        # width, height, fps should be numbers
        assert isinstance(cam["width"], (int, float))
        assert isinstance(cam["height"], (int, float))
        assert isinstance(cam["fps"], (int, float))

