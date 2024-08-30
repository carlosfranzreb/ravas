import logging
from copy import deepcopy
from functools import partial

import cv2 as cv
import sounddevice as sd
from PyQt6.QtWidgets import QDialogButtonBox, QFormLayout, QVBoxLayout, QComboBox, QMessageBox, QLabel

from .settings_helper import RestorableDialog


_logger = logging.getLogger('gui.config_dlg')


NO_SELECTION: str = '<no selection>'
""" 
a dummy label/item-data in case the current config-object has missing or invalid value, i.e. cannot be set to selected

i.e. this label/item-data indicates that the corresponding widget (or its underlying configuration-field) 
has an invalid value.
"""


class ConfigDialog(RestorableDialog):
    def __init__(self, parent, config: dict):
        """
        create modal configuration dialog

        :param parent: the parent window/gadget for the modal dialog
        :param config: the configuration to be shown / modified
                       __NOTE:__ will use a __copy__ of this in the configuration dialog; the (possibly)
                                 modified (copy) will be available in field `config` after the dialog is closed
        """
        super().__init__(parent=parent)
        self.setWindowTitle("Configuration")
        dialogLayout = QVBoxLayout()
        formLayout = QFormLayout()

        self.config = deepcopy(config)

        # # FIXME TEST some dummy settings:
        # formLayout.addRow("Name:", QLineEdit())
        # formLayout.addRow("Age:", QLineEdit())
        # formLayout.addRow("Job:", QLineEdit())
        # formLayout.addRow("Hobbies:", QLineEdit())

        cbbAudioIn = self._createAudioDeviceWidget(audio_input=True)
        formLayout.addRow("Audio Input:", cbbAudioIn)

        cbbAudioOut = self._createAudioDeviceWidget(audio_input=False)
        formLayout.addRow("Audio Output:", cbbAudioOut)

        # FIXME DISABLED takes too long, find better solution for generating video device list,
        #                other than trying out opencv2 indicies ... or maybe show fast solution first and give users
        #                the option to start a time consuming search, if they want to
        # cbbVideoIn = self._createVideoDeviceWidget()
        cbbVideoIn = QComboBox()  # FIXME dummy
        cbbVideoIn.addItem('<disabled>')  # FIXME dummy
        cbbVideoIn.setEnabled(False)  # FIXME dummy
        formLayout.addRow("Video Input:", cbbVideoIn)

        cbbVideoConverter = self._createVideoConverterWidget()
        formLayout.addRow("Video Converter:", cbbVideoConverter)

        cbbAvatar = self._createAvatarWidget()
        formLayout.addRow("Avatar:", cbbAvatar)

        # MOD cbbVideoConverter: enable/disable cbbAvatar when converter "Avatar" is selected/deselected
        def _updateAvatarEnabled(selected_video_converter: str):
            cbbAvatar.setEnabled(selected_video_converter == 'Avatar')
        _updateAvatarEnabled(cbbVideoConverter.currentText())
        cbbVideoConverter.currentTextChanged.connect(_updateAvatarEnabled)

        cbbAudioVoice = self._createAudioVoiceWidget()
        formLayout.addRow("Audio Voice:", cbbAudioVoice)

        # TODO add config widgets for:
        # use_audio: true
        #
        # use_video: true
        # output_window: true
        #
        #
        # log_level: INFO
        #
        # gui_log_level TODO
        # disable_console_logging   TODO
        #
        # ? [video]/[converter]
        # max_fps: 20
        # width: *video_width
        # height: *video_height

        dialogLayout.addLayout(formLayout)

        buttons = self._createDlgCtrls()
        dialogLayout.addWidget(buttons)
        self.setLayout(dialogLayout)

    # override RestorableDialog.getSettings():
    def getSettingsPath(self) -> str:
        return 'config_dlg'

    def _createDlgCtrls(self):
        buttons = QDialogButtonBox()
        buttons.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.reject)
        return buttons

    def _createAudioDeviceWidget(self, audio_input: bool) -> QComboBox:
        fieldName = 'input_device' if audio_input else 'output_device'
        return self._createTextComboBox(getAudioDevices(audio_input), ['audio', fieldName])

    def _createVideoDeviceWidget(self) -> QComboBox:
        indices = returnCameraIndices()  # FIXME takes too long, find better solution
        return self._createTextComboBox(indices, ['video', 'input_device'])

    def _createVideoConverterWidget(self) -> QComboBox:
        items = {
            'Avatar': 'stream_processing.models.Avatar',
            'FaceMask': 'stream_processing.models.FaceMask',
            'Echo': 'stream_processing.models.Echo',
        }
        return self._createDataComboBox(items, ['video', 'converter', 'cls'])

    def _createAvatarWidget(self) -> QComboBox:
        items = {
            'Avatar (Female)': './default_avatar_alt.glb',
            'Avatar (Male)': './default_avatar.glb',
        }
        return self._createDataComboBox(items, ['video', 'converter', 'avatar_uri'])

    def _createAudioVoiceWidget(self) -> QComboBox:
        items = {
            'Female (Wendy)': './target_feats/wendy.pt',
            'Male (John)': './target_feats/john.pt',
        }
        return self._createDataComboBox(items, ['audio', 'converter', 'target_feats_path'])

    def _createTextComboBox(self, text_items: list[str], config_path: list[str]) -> QComboBox:
        """
        HELPER create a combo-box with simple text-items:
        on selection the corresponding text-value will be applied to the `config_path`.

        :param text_items: the text-items; when selected their value will be applied to the config
        :param config_path: the path to the configuration value that will be updated
        :returns: the created combo-box
        """
        combobox = QComboBox()
        combobox.addItems(text_items)
        current_value, field_name, sub_config = self._get_current_value_and_config_path_for(config_path)
        # set current_value as selected index:
        try:
            idx = text_items.index(current_value)
            combobox.setCurrentIndex(idx)
        except:
            combobox.insertItem(0, NO_SELECTION)
            combobox.setCurrentIndex(0)
        combobox.currentTextChanged.connect(partial(self.setConfigValue, sub_config, field_name))
        return combobox

    def _createDataComboBox(self, item_data: dict[str, any], config_path: list[str]) -> QComboBox:
        """
        HELPER create combo-box with data-items:
        the `key` will be used as label in the combo-box, and the `value` will be applied  to the `config_path`, when
        the corresponding item is selected.

        :param item_data: the data-items, where the `key` is used as label and the `value` is applied to the config upon selection
        :param config_path: the path to the configuration value that will be updated
        :returns: the created combo-box
        """
        combobox = QComboBox()
        current_value, field_name, sub_config = self._get_current_value_and_config_path_for(config_path)
        idx = 0
        did_select_current = False
        for key, cls in item_data.items():
            combobox.addItem(key, cls)
            if cls == current_value:
                did_select_current = True
                combobox.setCurrentIndex(idx)
            idx += 1
        if not did_select_current:
            combobox.insertItem(0, NO_SELECTION, NO_SELECTION)
            combobox.setCurrentIndex(0)
        combobox.currentIndexChanged.connect(partial(self.setConfigValueByData, combobox, sub_config, field_name))
        return combobox

    def _get_current_value_and_config_path_for(self, config_path: list[str]) -> tuple[any, str, dict]:
        """
        HELPER for a path in the config return:
         * the `current_value` at that path
         * the last containing dictionary `sub_config` that contains the value
         * the `field_name` within that `sub_config` that refers to the `current_value`

        i.e. something like
        sub_config ~> self.config[...config_path[:-1]]
        current_value = sub_config[config_path[-1:]

        :param config_path: the access path in the config
        :returns: a `tuple[<current_value>, <field_name>, <sub_config>]`
        """
        current_value = None
        field_name = None
        sub_config = self.config
        for curr_field in config_path:
            field_name = curr_field
            current_value = sub_config.get(field_name)
            if isinstance(current_value, dict):
                sub_config = current_value
        return current_value, field_name, sub_config

    def setConfigValue(self, sub_config: dict, field: str, value: any):
        print('set config ', sub_config, ':', field, ' -> ', value)  # FIXME DEBUG
        if value != NO_SELECTION:
            sub_config[field] = value
        else:
            _logger.warning('selected NO_SELECTION for %s: reset config value to None!', field)
            sub_config[field] = None

    def setConfigValueByData(self, widget: QComboBox, sub_config: dict, field: str, idx: int):
        data = widget.itemData(idx)
        print('set config by index ', sub_config, ':', field, ' -> ComboBox[', idx, '] = [', data, ']')  # FIXME DEBUG
        if data != NO_SELECTION:
            sub_config[field] = data
        else:
            _logger.warning('selected NO_SELECTION for %s: reset config value to None!', field)
            sub_config[field] = None

    def validateAndAccept(self):
        has_invalid = False
        errors = {}

        # NOTE: iterate over ALL children of QFormLayout (instead of only ComboBoxes themselves) in order to access
        #       preceding labels for ComboBoxes, for creating details in error-messages in case of invalid values
        # boxes: list[QComboBox] = self.findChildren(QComboBox)

        formLayout: QFormLayout = self.layout().itemAt(0)
        for i in range(formLayout.count()):
            widget = formLayout.itemAt(i).widget()
            if isinstance(widget, QComboBox):
                # validate selection in combo-box:
                #   must not have NO_SELECTION as current selection
                sel = widget.currentText()
                if sel == NO_SELECTION:
                    has_invalid = True
                    did_add_error = False
                    if i > 0:
                        # get text from combo-box's (preceding) label
                        label = formLayout.itemAt(i-1).widget()
                        if label and isinstance(label, QLabel):
                            did_add_error = True
                            label_str = label.text()
                            # add colon, if not present (for error message formatting)
                            if ':' not in label_str:
                                label_str += ':'
                            errors[label_str] = sel
                    if not did_add_error:
                        errors['unknown'] = sel

        if not has_invalid:
            self.accept()
        else:
            details = ''
            for msg in errors:
                details += '\n * {} {}'.format(msg, errors[msg])
            QMessageBox.question(self, 'Invalid Value', 'Please select a different value(s) for ' + details, QMessageBox.StandardButton.Ok)


def getAudioDevices(is_input: bool) -> list[str]:
    """Retrieve the device names for audio-input or -output index"""
    devices = sd.query_devices()
    _logger.debug('audio devices: %s', devices)
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
def returnCameraIndices() -> list[str]:
    # checks the first 10 indexes.
    # WARNING: this function takes noticeable time to run
    #          (since it does actually open cameras & captures for testing if camera is usable)
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv.VideoCapture(index)

        if _logger.isEnabledFor(logging.DEBUG):
            camera_info = (' | backend: "%s" | exception mode: %s' % (cap.getBackendName(), cap.getExceptionMode()) if cap.isOpened() else '')
            _logger.debug('check camera %s -> opened: %s%s', index, cap.isOpened(), camera_info)

        if cap.isOpened() and cap.read()[0]:
            arr.append(str(index))
            cap.release()
        index += 1
        i -= 1
    return arr
