import logging
from functools import partial
from typing import Callable, NoReturn, Union, Optional

from .config_utils import get_audio_devices, return_camera_indices, get_voices_from

_logger = logging.getLogger('gui.config_items')


NO_SELECTION: str = '<no selection>'
""" 
a dummy label/item-data in case the current config-object has missing or invalid value, i.e. cannot be set to selected

i.e. this label/item-data indicates that the corresponding widget (or its underlying configuration-field) 
has an invalid value.
"""


class ConfigurationItem:
    """
    class for specifying configuration items:
    an access path (in the configuration dictionary) and a list of possible, valid configuration-values
    (i.e. that can be safely applied to the configuration).

    The configuration-values are either a (string) list, or a dictionary of labels and configuration values.
    Alternatively, a generator-function can be supplied that creates the configuration-values.

    The configuration-values should be accessed either by `ConfigurationItem.get()` or by
    `ConfigurationItem.get_latest()`; the later one will update the values with the generator-function,
    if one was supplied.
    """
    def __init__(self, configuration_path: list[str], configuration_values: list[str] | dict[str, any] | Callable[[Union[list[str], dict[str, any]]], NoReturn]):
        self.config_path: list[str] = configuration_path
        if callable(configuration_values):
            self.create_values: Optional[Callable[[Union[list[str], dict[str, any]]], NoReturn]] = configuration_values
            self.config_values: Optional[Union[list[str], dict[str, any]]] = None
        else:
            self.create_values: Optional[Callable[[Union[list[str], dict[str, any]]], NoReturn]] = None
            self.config_values: Optional[Union[list[str], dict[str, any]]] = configuration_values

    def get(self):
        if not self.config_values and self.create_values:
            return self.get_latest()
        return self.config_values

    def get_latest(self):
        if self.create_values:
            self.config_values = self.create_values()
        return self.config_values


CONFIG_ITEMS: dict[str, ConfigurationItem] = {
    'audio_input_devices': ConfigurationItem(['audio', 'input_device'],
                                             partial(get_audio_devices, is_input=True, logger=_logger)),

    'audio_output_devices': ConfigurationItem(['audio', 'output_device'],
                                              partial(get_audio_devices, is_input=False, logger=_logger)),

    'video_converters': ConfigurationItem(['video', 'converter', 'cls'], {
        'Avatar':                   'stream_processing.models.Avatar',
        'FaceMask':                 'stream_processing.models.FaceMask',
        'Echo (No Anonymization)':  'stream_processing.models.Echo',
    }),

    'video_avatars': ConfigurationItem(['video', 'converter', 'avatar_uri'], {
        'Avatar (Female)':      'avatar_1_f.glb',
        'Avatar (Male)':        'avatar_2_m.glb',
        'Avatar 2 (Female)':    'avatar_3_f.glb',
        'Avatar 2 (Male)':      'avatar_4_m.glb',
    }),

    'audio_voices': ConfigurationItem(['audio', 'converter', 'target_feats_path'],
                                      partial(get_voices_from, dir_path='./target_feats', logger=_logger)),

    'log_levels': ConfigurationItem(['log_level'], {
        '<DEFAULT>':    'INFO',
        'CRITICAL':     'CRITICAL',
        'ERROR':        'ERROR',
        'WARN':         'WARNING',
        'INFO':         'INFO',
        'DEBUG':        'DEBUG',
    }),

    'gui_log_levels': ConfigurationItem(['gui_log_level'], {
        '<DEFAULT>':    None,
        'CRITICAL':     'CRITICAL',
        'ERROR':        'ERROR',
        'WARN':         'WARNING',
        'INFO':         'INFO',
        'DEBUG':        'DEBUG',
    }),

    # FIXME current implementation of `return_camera_indices()` takes too long -> find better solution
    'video_input_devices': ConfigurationItem(['video', 'input_device'],
                                             partial(return_camera_indices, logger=_logger)),
}
""" definitions of configuration-items that should be configurable by users in GUI """


IGNORE_CONFIG_ITEM_KEYS = {'video_input_devices'}  # FIXME remove entry 'video_input_devices': should also handle video input-device, when retrieving valid list is more performant
"""
configuration-item keys (in `CONFIG_ITEMS`) that should currently be ignored (e.g. due to performance issues)
"""