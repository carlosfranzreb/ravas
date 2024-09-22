import logging
from functools import partial
from typing import Callable, Union, Optional, Literal

from .config_utils import get_audio_devices, get_camera_device_items, get_voices_from, \
    get_current_value_and_config_path_for
from ..utils import resolve_file_path


_logger = logging.getLogger('gui.config_items')


NO_SELECTION: str = '<no selection>'
""" 
a dummy label/item-data in case the current config-object has missing or invalid value, i.e. cannot be set to selected

i.e. this label/item-data indicates that the corresponding widget (or its underlying configuration-field) 
has an invalid value.
"""

AVATAR_CONVERTER: str = 'stream_processing.models.Avatar'
""" for reference: the configuration value for selecting the avatar-converter (video anonymization) """


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
    def __init__(self,
                 configuration_path: list[str],
                 configuration_values: list[str] | dict[str, any] | Callable[[], Union[list[str], dict[str, any]]],
                 is_ignore_validation: Optional[Callable[[dict], bool]] = None
        ):
        self.config_path: list[str] = configuration_path
        self.is_ignore_validation: Optional[Callable[[dict], bool]] = is_ignore_validation
        if callable(configuration_values):
            self.create_values: Optional[Callable[[], Union[list[str], dict[str, any]]]] = configuration_values
            self.config_values: Optional[Union[list[str], dict[str, any]]] = None
        else:
            self.create_values: Optional[Callable[[], Union[list[str], dict[str, any]]]] = None
            self.config_values: Optional[Union[list[str], dict[str, any]]] = configuration_values

    def get(self):
        if not self.config_values and self.create_values:
            return self.get_latest()
        return self.config_values

    def get_latest(self):
        if self.create_values:
            self.config_values = self.create_values()
        return self.config_values

    def can_ignore_validation(self, current_config: dict) -> bool:
        if not self.is_ignore_validation:
            return False
        return self.is_ignore_validation(current_config)


CONFIG_ITEMS: dict[str, ConfigurationItem] = {

    'use_audio': ConfigurationItem(['audio', 'use_audio'], {
        'Use Audio':        True,
        'Disable Audio':    False,
    }),

    'use_video': ConfigurationItem(['video', 'use_video'], {
        'Use Video':        True,
        'Disable Video':    False,
    }),

    'audio_input_devices': ConfigurationItem(['audio', 'input_device'],
                                             partial(get_audio_devices, is_input=True, logger=_logger)),

    'audio_output_devices': ConfigurationItem(['audio', 'output_device'],
                                              partial(get_audio_devices, is_input=False, logger=_logger)),

    'video_converters': ConfigurationItem(['video', 'converter', 'cls'], {
        'Avatar':                   AVATAR_CONVERTER,
        'FaceMask':                 'stream_processing.models.FaceMask',
        'Echo (No Anonymization)':  'stream_processing.models.Echo',
    }),

    'video_avatars': ConfigurationItem(['video', 'converter', 'avatar_uri'], {
        'Avatar (Female)':      'avatar_1_f.glb',
        'Avatar (Male)':        'avatar_2_m.glb',
        'Avatar 2 (Female)':    'avatar_3_f.glb',
        'Avatar 2 (Male)':      'avatar_4_m.glb',
    }),

    'output_window': ConfigurationItem(['video', 'output_window'], {
        'Show Video Output Window': True,
        'Do Not Show Output Video': False,
    }),

    'audio_voices': ConfigurationItem(['audio', 'converter', 'target_feats_path'],
                                      partial(get_voices_from, dir_path=resolve_file_path('target_feats/'), logger=_logger)),

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

    'disable_console_logging': ConfigurationItem(['disable_console_logging'], {
        '<DEFAULT> (Disable)': None,
        'Disable Logging To Console': True,
        'Enable Logging To Console': False,
    }),


    # FIXME current implementation of `return_camera_indices()` takes too long -> find better solution
    'video_input_devices': ConfigurationItem(['video', 'input_device'],
                                             partial(get_camera_device_items, logger=_logger)),
}
""" definitions of configuration-items that should be configurable by users in GUI """


IGNORE_CONFIG_ITEM_KEYS = {'video_input_devices'}  # FIXME remove entry 'video_input_devices': should also handle video input-device, when retrieving valid list is more performant
"""
configuration-item keys (in `CONFIG_ITEMS`) that should currently be ignored (e.g. due to performance issues)
"""


def is_media_disabled(media: Union[Literal['audio'], Literal['video']], config: dict) -> bool:
    field = 'use_' + media
    val, _, _ = get_current_value_and_config_path_for(config, [media, field])
    return val is False


def _do_set_ignore_validation_helpers():
    """
    HELPER: initialize & set helper function for config-items that indicate, if their validation can be ignored
            (based on the config-values at that time)
    """
    # audio-configs that can be ignored, if audio is disabled:
    CONFIG_ITEMS['audio_input_devices'].is_ignore_validation = partial(is_media_disabled, 'audio')
    CONFIG_ITEMS['audio_voices'].is_ignore_validation = partial(is_media_disabled, 'audio')
    CONFIG_ITEMS['audio_output_devices'].is_ignore_validation = partial(is_media_disabled, 'audio')

    # video-configs that can be ignored, if video is disabled:
    CONFIG_ITEMS['video_input_devices'].is_ignore_validation = partial(is_media_disabled, 'video')
    CONFIG_ITEMS['video_converters'].is_ignore_validation = partial(is_media_disabled, 'video')
    CONFIG_ITEMS['output_window'].is_ignore_validation = partial(is_media_disabled, 'video')

    # custom (i.e. more complicated) config-settings that depend on other / multiple other config-values

    def can_ignore_avatar_validation(config: dict):
        if is_media_disabled('video', config):
            return True
        val, _, _ = get_current_value_and_config_path_for(config, CONFIG_ITEMS['video_avatars'].config_path)
        return val != AVATAR_CONVERTER

    CONFIG_ITEMS['video_avatars'].is_ignore_validation = can_ignore_avatar_validation


# do apply ignore-validation helpers for config-items:
_do_set_ignore_validation_helpers()
