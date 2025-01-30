import logging
from functools import partial
from inspect import signature, Parameter
from typing import Callable, Union, Optional, Literal

from .config_utils import get_audio_devices, get_camera_device_items, get_voices_from, \
    get_current_value_and_config_path_for, is_port_valid, is_positive_number, wrap_simple_validate, \
    is_positive_number_and_equal_to_config_path, get_avatars_for_config
from ..models.avatar.avatar import RenderAppType
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

    1. (infinite) Generative configuration values:

    In case the configuration is infinite (or would require a large list/dictionary), and setting the
    configuration-value is done with an input field (e.g. for entering strings or numbers), the ConfigurationItem
    should be set with a validator function `is_valid_value(value, config | None) -> bool`.


    2. (finite) Closed-list configuration values:

    The configuration-values are either a (string) list, or a dictionary of labels and configuration values.
    Alternatively, a generator-function can be supplied that creates the configuration-values.

    The configuration-values should be accessed either by `ConfigurationItem.get()` or by
    `ConfigurationItem.get_latest()`; the later one will update the values with the generator-function,
    if one was supplied.
    """
    def __init__(self,
                 configuration_path: list[str],
                 configuration_values: list[str] | dict[str, any] | Callable[[], Union[list[str], dict[str, any]]] | Callable[[Optional[dict]], Union[list[str], dict[str, any]]] | None,
                 is_ignore_validation: Optional[Callable[[dict], bool]] = None,
                 is_valid_value: Optional[Callable[[any, Optional[dict]], bool]] = None
    ):
        """

        :param configuration_path: the "path" to the configuration property within the YAML configuration structure
                                   (each entry in the path list represents 1 hierarchy level, the last being the
                                    property name itself, e.g. `["video", "use_video"]`)
        :param configuration_values: the allowed configuration values (also see class comment on more details)
        :param is_ignore_validation: OPTIONAL helper function that will indicate, if based on the current configuration
                                     values this configuration item can be ignored
                                     (i.e. does not require to have a valid value at the moment):  \
                                     `is_ignore_validation(current_config: Dict) -> bool`
        :param is_valid_value: OPTIONAL helper function for validating, if the configuration item currently has a valid
                               value: by default (i.e. if omitted) a valid value will be determined by using the
                               `configuration_values` parameter; if validation needs special processing (e.g. depends on
                               other configuration values), it can be implemented by using & setting this helper method:  \
                               `is_valid_value(current_config: Dict) -> bool`
        """
        self.config_path: list[str] = configuration_path
        self.is_ignore_validation: Optional[Callable[[dict], bool]] = is_ignore_validation
        self.is_valid_value: Optional[Callable[[any, Optional[dict]], bool]] = is_valid_value
        self._create_values_has_no_args: Optional[bool] = None
        if callable(configuration_values):
            self.create_values: Optional[Callable[[], Union[list[str], dict[str, any]]]] | Optional[Callable[[Optional[dict]], Union[list[str], dict[str, any]]]] = configuration_values
            self.config_values: Optional[Union[list[str], dict[str, any]]] = None
        else:
            self.create_values: Optional[Callable[[], Union[list[str], dict[str, any]]]] | Optional[Callable[[Optional[dict]], Union[list[str], dict[str, any]]]] = None
            self.config_values: Optional[Union[list[str], dict[str, any]]] = configuration_values

    @staticmethod
    def _has_non_default_or_named_config_arg(func: Callable) -> bool:
        """ helper to determine, if a `create_values()` function has the (optional) config-parameter or not """
        for p in signature(func).parameters.values():
            # heuristic: if a parameter has name "config", then TRUE
            if p.name == 'config':
                return True
            # heuristic: if there is any parameter that has NO default value set, then TRUE
            if p.default == Parameter.empty:
                return True
        return False

    @property
    def create_values(self) -> Optional[Callable[[], Union[list[str], dict[str, any]]]] | Optional[Callable[[Optional[dict]], Union[list[str], dict[str, any]]]]:
        return self._create_values

    @create_values.setter
    def create_values(self, value: Optional[Callable[[], Union[list[str], dict[str, any]]]] | Optional[Callable[[Optional[dict]], Union[list[str], dict[str, any]]]]):
        self._create_values = value
        self._create_values_has_no_args = not self._has_non_default_or_named_config_arg(value) if value else None

    def get(self, config: Optional[dict] = None):
        if not self.config_values and self.create_values:
            return self.get_latest(config)
        return self.config_values

    def get_latest(self, config: Optional[dict] = None):
        if self.create_values:
            if self._create_values_has_no_args:
                self.config_values = self.create_values()
            else:
                self.config_values = self.create_values(config)
        return self.config_values

    def can_ignore_validation(self, current_config: dict) -> bool:
        if not self.is_ignore_validation:
            return False
        return self.is_ignore_validation(current_config)


# prepare configuration-items for video-height that require inter-dependent validation:
config_item_video_output_height = ConfigurationItem(
    ['video', 'height'], None)  # omit validation here & "back-reference", see below
config_item_video_converter_height = ConfigurationItem(
    ['video', 'converter', 'height'], None,
    is_valid_value=partial(is_positive_number_and_equal_to_config_path, config_path=config_item_video_output_height.config_path))
# no set "back-reference" validation against config_item_video_converter_height:
config_item_video_output_height.is_valid_value = partial(is_positive_number_and_equal_to_config_path,
                                                         config_path=config_item_video_converter_height.config_path)

# prepare configuration-items for video-width that require inter-dependent validation:
config_item_video_output_width = ConfigurationItem(
    ['video', 'width'], None)  # omit validation here & "back-reference", see below
config_item_video_converter_width = ConfigurationItem(
    ['video', 'converter', 'width'], None,
    is_valid_value=partial(is_positive_number_and_equal_to_config_path, config_path=config_item_video_output_width.config_path))
# no set "back-reference" validation against config_item_video_converter_width:
config_item_video_output_width.is_valid_value = partial(is_positive_number_and_equal_to_config_path,
                                                        config_path=config_item_video_converter_width.config_path)


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
    'video_avatars': ConfigurationItem(['video', 'converter', 'avatar_uri'],
                                       partial(get_avatars_for_config, logger=_logger)),

    'avatar_renderer': ConfigurationItem(['video', 'converter', 'avatar_renderer'], {
        'OpenGL App (Default)': RenderAppType.OPENGL_APP.value,
        'Browser (Chrome)':     RenderAppType.BROWSER.value,
    }),

    'avatar_ws_port': ConfigurationItem(['video', 'converter', 'ws_port'], None,
                                        is_valid_value=wrap_simple_validate(is_port_valid)),

    'output_window': ConfigurationItem(['video', 'output_window'], {
        'Show Video Output Window': True,
        'Do Not Show Output Video': False,
    }),

    'avatar_render_window': ConfigurationItem(['video', 'converter', 'show_renderer_window'], {
        'Show Avatar Renderer Window':        True,
        'Do Not Show Avatar Renderer Window': False,
    }),

    # DISABLED selecting virtual-camera backend via combo-box (only enabled/disabled check-box for now):
    # 'output_virtual_cam': ConfigurationItem(['video', 'output_virtual_cam'], get_virtual_camera_backends),

    'output_virtual_cam': ConfigurationItem(['video', 'output_virtual_cam'], {
        'Enable Virtual Camera Output': True,
        'Disable Virtual Camera Output': False,
    }),

    'video_output_height': config_item_video_output_height,
    'video_converter_height': config_item_video_converter_height,
    'video_output_width': config_item_video_output_width,
    'video_converter_width': config_item_video_converter_width,

    'video_output_fps': ConfigurationItem(['video', 'max_fps'], None,
                                          is_valid_value=wrap_simple_validate(is_positive_number)),

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
    CONFIG_ITEMS['output_virtual_cam'].is_ignore_validation = partial(is_media_disabled, 'video')

    CONFIG_ITEMS['video_output_fps'].is_ignore_validation = partial(is_media_disabled, 'video')
    CONFIG_ITEMS['video_output_width'].is_ignore_validation = partial(is_media_disabled, 'video')
    CONFIG_ITEMS['video_converter_width'].is_ignore_validation = partial(is_media_disabled, 'video')
    CONFIG_ITEMS['video_output_height'].is_ignore_validation = partial(is_media_disabled, 'video')
    CONFIG_ITEMS['video_converter_height'].is_ignore_validation = partial(is_media_disabled, 'video')

    # custom (i.e. more complicated) config-settings that depend on other / multiple other config-values

    def can_ignore_avatar_validation(config: dict):
        if is_media_disabled('video', config):
            return True
        val, _, _ = get_current_value_and_config_path_for(config, CONFIG_ITEMS['video_converters'].config_path)
        return val != AVATAR_CONVERTER

    def can_ignore_browser_avatar_validation(config: dict):
        if can_ignore_avatar_validation(config):
            return True
        val, _, _ = get_current_value_and_config_path_for(config, CONFIG_ITEMS['avatar_renderer'].config_path)
        return val != RenderAppType.BROWSER.value

    CONFIG_ITEMS['video_avatars'].is_ignore_validation = can_ignore_avatar_validation
    CONFIG_ITEMS['avatar_renderer'].is_ignore_validation = can_ignore_avatar_validation
    CONFIG_ITEMS['avatar_render_window'].is_ignore_validation = can_ignore_avatar_validation
    CONFIG_ITEMS['avatar_ws_port'].is_ignore_validation = can_ignore_browser_avatar_validation


# do apply ignore-validation helpers for config-items:
_do_set_ignore_validation_helpers()
