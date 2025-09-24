import json
import logging
import re
from copy import deepcopy
from functools import partial
from typing import Tuple, Callable, Optional

import yaml
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
    QComboBox,
    QMessageBox,
    QLabel,
    QGroupBox,
    QSizePolicy,
    QWidget,
    QFileDialog,
    QCheckBox,
    QPushButton,
    QHBoxLayout,
    QApplication,
    QSpinBox,
    QSlider,
    QTabWidget,
    QScrollArea,
)

from .config_items import (
    CONFIG_ITEMS,
    NO_SELECTION,
    ConfigurationItem,
    AVATAR_CONVERTER,
)
from .config_utils import get_current_value_and_config_path_for, validate_config_values
from .settings_helper import (
    RestorableDialog,
    RestoreSettingItem,
    StoreSettingItem,
    applySetting,
)
from ..models.avatar.avatar import RenderAppType


_logger = logging.getLogger("gui.config_dlg")


STYLE_INVALID_SELECTION = "color: red"
""" widget style to indicate an invalid selection/configuration """

CONFIG_PATH_SEP = "/"
"""
separator for config-keys when "flattening" the config / config-paths

EXAMPLE:
```
{"some": {"sub": "key"}}
```
with flattened config / path (using `/` as separator):
```
{"some/sub": "key"}}
```
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

        self.isResetConfig = False
        """ will be set to `True` if user request reset for config """
        self.config = deepcopy(config)
        """ (deep copy of) the configuration """
        self.changed_config: dict = {}
        """ 
        flattened dictionary containing changed config values (changed w.r.t. to the original config-file)
        (stored/restored from settings; the keys are "flattened" using `CONFIG_PATH_SEP` as separator)
        """

        # do restore config-changes from settings and apply them:
        #   (load them "flattened" into self.changed_config, and apply them to self.config)
        self._restoreAndApplyConfigChangeSettings()

        mainSettingsLayout = QVBoxLayout()

        # ############ INPUT ################
        inputForm = QFormLayout()

        cbbAudioIn = self._createComboBoxFor(CONFIG_ITEMS["audio_input_devices"])
        inputForm.addRow("Audio Input:", cbbAudioIn)

        videoInLayout, cbbVideoIn, btnDetectVideoIn = (
            self._createVideoInputDeviceWidget()
        )
        inputForm.addRow("Video Input:", videoInLayout)

        inputGroup = self._makeGroupBox("Input", inputForm)
        mainSettingsLayout.addWidget(inputGroup)

        # ############ OUTPUT ################
        outputForm = QFormLayout()

        # cbbAudioOut = self._createAudioDeviceWidget(audio_input=False)
        cbbAudioOut = self._createComboBoxFor(CONFIG_ITEMS["audio_output_devices"])
        outputForm.addRow("Audio Output:", cbbAudioOut)

        # DISABLED selecting virtual-camera backend via combo-box (for now, use check-box for enabling/disabling):
        # cbbEnableVirtualCamera = self._createComboBoxFor(CONFIG_ITEMS['output_virtual_cam'])
        chkEnableVirtualCamera = self._createCheckBoxFor(
            CONFIG_ITEMS["output_virtual_cam"]
        )
        layoutVirtualCamera = QVBoxLayout()
        layoutVirtualCamera.addWidget(chkEnableVirtualCamera)
        layoutVirtualCamera.addWidget(
            QLabel(
                '(requires additional software like "OBS Virtual Camera" or "Unity Video Capture")'
            )
        )
        outputForm.addRow("Enable Virtual Camera Output:", layoutVirtualCamera)

        sldOutputFramerate, layoutOutputFramerate = self._createSliderFor(
            CONFIG_ITEMS["video_output_fps"], min=10, max=60, step=1
        )
        outputForm.addRow("Output Video Frame Rate:", layoutOutputFramerate)

        sldOutputVideoWidth, layoutOutputVideoWidth = self._createSliderFor(
            CONFIG_ITEMS["video_output_width"],
            min=360,
            max=8192,
            step=16,
            additional_config_path=CONFIG_ITEMS["video_converter_width"].config_path,
        )
        outputForm.addRow("Output Video Width:", layoutOutputVideoWidth)

        sldOutputVideoHeight, layoutOutputVideoHeight = self._createSliderFor(
            CONFIG_ITEMS["video_output_height"],
            min=288,
            max=4096,
            step=16,
            additional_config_path=CONFIG_ITEMS["video_converter_height"].config_path,
        )
        outputForm.addRow("Output Video Height:", layoutOutputVideoHeight)

        outputGroup = self._makeGroupBox("Output", outputForm)
        mainSettingsLayout.addWidget(outputGroup)

        # ############ CONVERTER AUDIO ################

        convertAudioForm = QFormLayout()

        cbbAudioAnonymizer = self._createComboBoxFor(CONFIG_ITEMS["anonymizers"])
        convertAudioForm.addRow("Voice anonymizer:", cbbAudioAnonymizer)

        cbbAudioVoice = self._createComboBoxFor(CONFIG_ITEMS["audio_voices"])
        convertAudioForm.addRow("Audio Voice:", cbbAudioVoice)

        convertAudioGroup = self._makeGroupBox("Convert Audio", convertAudioForm)
        mainSettingsLayout.addWidget(convertAudioGroup)

        # ############ CONVERTER VIDEO ################
        convertVideoForm = QFormLayout()

        # cbbVideoConverter = self._createVideoConverterWidget()
        cbbVideoConverter = self._createComboBoxFor(CONFIG_ITEMS["video_converters"])
        convertVideoForm.addRow("Video Converter:", cbbVideoConverter)

        # cbbAvatar = self._createAvatarWidget()
        cbbAvatar = self._createComboBoxFor(CONFIG_ITEMS["video_avatars"])
        convertVideoForm.addRow("Avatar:", cbbAvatar)

        # NOTE this will be added to ADVANCED SETTINGS group (see below), but we need it here
        #      to enable/disable if avatar-converter is selected
        cbbAvatarRenderer = self._createComboBoxFor(CONFIG_ITEMS["avatar_renderer"])

        # NOTE this will be added to ADVANCED SETTINGS group (see below), but we need it here
        #      to enable/disable if avatar-converter is selected
        chkShowAvatarRendererWindow = self._createCheckBoxFor(
            CONFIG_ITEMS["avatar_render_window"]
        )

        # NOTE this will be added to ADVANCED SETTINGS group (see below), but we need it here
        #      to enable/disable if avatar-converter is selected
        iptAvatarPort, lbAvatarPort, containerAvatarPort = self._createPortInput()

        # MOD cbbVideoConverter: enable/disable cbbAvatar when converter "Avatar" is selected/deselected
        def _updateAvatarEnabled(selected_video_converter: str):
            enable = selected_video_converter == "Avatar"
            cbbAvatar.setEnabled(enable)
            cbbAvatarRenderer.setEnabled(enable)
            iptAvatarPort.setEnabled(enable)
            lbAvatarPort.setEnabled(enable)

        _updateAvatarEnabled(
            cbbVideoConverter.currentText()
        )  # <- update for current config-value
        cbbVideoConverter.currentTextChanged.connect(
            _updateAvatarEnabled
        )  # <- update for config-changes

        convertVideoGroup = self._makeGroupBox("Convert Video", convertVideoForm)
        mainSettingsLayout.addWidget(convertVideoGroup)

        ########################################################################

        advancedSettingsLayout = QVBoxLayout()

        # ############ ADVANCED AUDIO / VIDEO SETTINGS ################
        advancedSettingsForm = QFormLayout()

        chkShowVideoWindow = self._createCheckBoxFor(CONFIG_ITEMS["output_window"])
        advancedSettingsForm.addRow(
            "Show Video Output Window (DEBUG):", chkShowVideoWindow
        )

        # NOTE cbbAvatarRenderer was created before, so it can be used in _updateAvatarEnabled()
        advancedSettingsForm.addRow("Avatar Renderer:", cbbAvatarRenderer)

        # NOTE chkShowAvatarRendererWindow was created before, so it can be used in _updateAvatarEnabled()
        advancedSettingsForm.addRow(
            "Show Avatar Renderer Window (DEBUG):", chkShowAvatarRendererWindow
        )

        # add port-number-widget here (created above at video-converter-selection widget):
        advancedSettingsForm.addRow(
            "Avatar Converter Port (Browser Renderer):", containerAvatarPort
        )

        # MOD cbbAvatarRenderer: enable/disable iptAvatarPort when renderer is not browser/Chrome
        def _updateAvatarRendererSelected(selected_avatar_renderer: str):
            enable = "browser" in selected_avatar_renderer.lower()
            containerAvatarPort.setEnabled(enable)
            iptAvatarPort.setEnabled(enable)
            lbAvatarPort.setEnabled(enable)

        _updateAvatarRendererSelected(
            cbbAvatarRenderer.currentText()
        )  # <- update for current config-value
        cbbAvatarRenderer.currentTextChanged.connect(
            _updateAvatarRendererSelected
        )  # <- update for config-changes

        chkUseAudio = self._createCheckBoxFor(CONFIG_ITEMS["use_audio"])
        advancedSettingsForm.addRow("Use Audio:", chkUseAudio)

        def _set_audio_widgets_enabled(_value):
            enable: bool = chkUseAudio.checkState() == QtCore.Qt.CheckState.Checked
            cbbAudioIn.setEnabled(enable)
            cbbAudioOut.setEnabled(enable)
            cbbAudioAnonymizer.setEnabled(enable)
            cbbAudioVoice.setEnabled(enable)

        _set_audio_widgets_enabled(chkUseAudio)  # <- update for current config
        chkUseAudio.stateChanged.connect(
            _set_audio_widgets_enabled
        )  # <- update on config changes

        chkUseVideo = self._createCheckBoxFor(CONFIG_ITEMS["use_video"])
        advancedSettingsForm.addRow("Use Video:", chkUseVideo)

        def _set_video_widgets_enabled(_value):
            enable: bool = chkUseVideo.checkState() == QtCore.Qt.CheckState.Checked
            # cbbEnableVirtualCamera.setEnabled(enable)
            chkEnableVirtualCamera.setEnabled(enable)
            btnDetectVideoIn.setEnabled(enable)
            cbbVideoConverter.setEnabled(enable)
            chkShowVideoWindow.setEnabled(enable)
            sldOutputFramerate.setEnabled(enable)
            sldOutputVideoWidth.setEnabled(enable)
            sldOutputVideoHeight.setEnabled(enable)
            enable_video_in = enable
            enable_avatar = enable
            if enable:
                # enable video-input-device selection, if the combo-box has real items to select
                # (i.e. count >= 1, or if it's 1, is must not be the NO_SELECTION entry)
                if cbbVideoIn.count() == 0:
                    enable_video_in = False
                elif cbbVideoIn.count() == 1:
                    video_in_data = cbbVideoIn.itemData(0)
                    enable_video_in = video_in_data != NO_SELECTION
                # enable avatar-selection, if video-converter is set to avatar converter
                conv_val, _, _ = get_current_value_and_config_path_for(
                    self.config, CONFIG_ITEMS["video_converters"].config_path
                )
                enable_avatar = conv_val == AVATAR_CONVERTER
            cbbVideoIn.setEnabled(enable_video_in)
            cbbAvatar.setEnabled(enable_avatar)
            cbbAvatarRenderer.setEnabled(enable_avatar)
            iptAvatarPort.setEnabled(enable_avatar)
            lbAvatarPort.setEnabled(enable_avatar)

        _set_video_widgets_enabled(chkUseVideo)  # <- update for current config
        chkUseVideo.stateChanged.connect(
            _set_video_widgets_enabled
        )  # <- update on config changes

        enableAVGroup = self._makeGroupBox("Advanced Settings", advancedSettingsForm)
        advancedSettingsLayout.addWidget(enableAVGroup)

        # ############ LOG SETTINGS ################
        loggingForm = QFormLayout()

        # cbbMainLogging = self._createLogLevelWidget(for_gui=False)
        cbbMainLogging = self._createComboBoxFor(CONFIG_ITEMS["log_levels"])
        loggingForm.addRow("Logging:", cbbMainLogging)

        # cbbGuiLogging = self._createLogLevelWidget(for_gui=True)
        cbbGuiLogging = self._createComboBoxFor(CONFIG_ITEMS["gui_log_levels"])
        loggingForm.addRow("Logging (GUI):", cbbGuiLogging)

        # FIXME [russa] could/should be a check-box, but since default value is TRUE (if omitted / not set),
        #       it would be complicated to convert/apply in all necessary places
        #       ... so for now, for ease of implementation, use combo-box for this
        cbbDisableConsoleLogging = self._createComboBoxFor(
            CONFIG_ITEMS["disable_console_logging"]
        )
        loggingForm.addRow("Disable Logging To Console:", cbbDisableConsoleLogging)

        loggingGroup = self._makeGroupBox("Logging", loggingForm)
        advancedSettingsLayout.addWidget(loggingGroup)

        # ############ Dialog Layout & Controls ################

        mainSettingsWidget = QWidget()
        mainSettingsWidget.setLayout(mainSettingsLayout)
        mainSettingsScroll = QScrollArea()
        mainSettingsScroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        mainSettingsScroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        mainSettingsScroll.setWidgetResizable(True)
        mainSettingsScroll.setWidget(mainSettingsWidget)

        advancedSettingsWidget = QWidget()
        advancedSettingsWidget.setLayout(advancedSettingsLayout)
        advancedSettingsScroll = QScrollArea()
        advancedSettingsScroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        advancedSettingsScroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        advancedSettingsScroll.setWidgetResizable(True)
        advancedSettingsScroll.setWidget(advancedSettingsWidget)

        tabs = QTabWidget()
        tabs.addTab(mainSettingsScroll, "Settings")
        tabs.addTab(advancedSettingsScroll, "Advanced Settings")

        self.is_first_tab_change = True

        def on_tab_changed():
            # HACK: for some reason, the disabling due to cbbAvatarRenderer's selection is not updated
            #       in the advanced tab on the first time ... force selection change on first tab change
            #       so that the attached disable-updater function gets triggered when it becomes visible
            if self.is_first_tab_change:
                curr_idx = cbbAvatarRenderer.currentIndex()
                cbbAvatarRenderer.setCurrentIndex(-1)
                cbbAvatarRenderer.setCurrentIndex(curr_idx)
                self.is_first_tab_change = False
            self._forceAlignRowLabels()

        # NOTE cannot align labels in invisible tab(s), since they will all have the tab's width
        #      -> recalculate alignment when tab becomes visible
        tabs.currentChanged.connect(on_tab_changed)

        dialogLayout = QVBoxLayout()
        dialogLayout.addWidget(tabs)

        buttons = self._createDlgCtrls()
        dialogLayout.addWidget(buttons)

        self.setLayout(dialogLayout)

    def _makeGroupBox(self, title: str, layout: QFormLayout):
        group = QGroupBox(title)
        group.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        )  # prevent vertical growth
        group.setLayout(layout)
        return group

    # override RestorableDialog.getSettings():
    def getSettingsPath(self) -> str:
        return "config_dlg"

    def getStoreSettingsItems(self) -> list[StoreSettingItem]:
        # NOTE: only override getStoreSettingsItems(), and not getRestoreSettingsItems():
        #       the config-changes need to be restored before showing the dialog, within the constructor,
        #       so simply overriding getRestoreSettingsItems() would apply them too late
        #       -> see _restoreAndApplyConfigChangeSettings() for loading & applying the config-changes
        settings_path = self.getSettingsPath()
        settings = super().getStoreSettingsItems()
        settings.append(
            StoreSettingItem(settings_path + "/configChanges", self._storeConfigChanges)
        )
        return settings

    def showEvent(self, evt):
        super().showEvent(evt)
        self._forceAlignRowLabels()

    def closeEvent(self, evt):
        # if close-button on window is used:
        # the configuration dialog was canceled
        # -> discard config-changes BEFORE handling close-event in parent-implementation
        #    (e.g. before storing to user settings9
        self._discardConfigChanges(reset_stored_changes=False)
        super().closeEvent(evt)

    def _restoreAndApplyConfigChangeSettings(self):
        settings = self.getSettings()
        c = RestoreSettingItem(
            self.getSettingsPath() + "/configChanges", self._applyConfigChanges
        )
        applySetting(settings, c.field, c.apply_func, c.covert_func)

    def _applyConfigChanges(self, json_string: str):
        _logger.debug("applying config changes: %s", json_string)
        try:
            self.changed_config: dict = json.loads(json_string)
        except Exception as exc:
            _logger.error(
                'failed to restore configuration changes: could not changes parse as JSON -> "%s"',
                json_string,
                exc_info=exc,
            )
            self.changed_config = {}
        _logger.debug("applying config changes: %s", self.changed_config)
        for path_str, changed_value in self.changed_config.items():
            config_path = path_str.split(CONFIG_PATH_SEP)
            curr_val, field, sub_config = get_current_value_and_config_path_for(
                self.config, config_path
            )
            # _logger.debug('  applying change ', config_path, ' = ', curr_val, ' <- ', changed_value)
            sub_config[field] = changed_value

    def _storeConfigChanges(self) -> str:
        json_string = json.dumps(self.changed_config)
        _logger.debug("storing config changes: %s", json_string)
        return json_string

    def _discardConfigChanges(self, reset_stored_changes: bool):
        self.changed_config.clear()
        if not reset_stored_changes:
            self._restoreAndApplyConfigChangeSettings()

    def _forceAlignRowLabels(self):
        """
        HELPER set all label's minimum width to the maximum of the currently displayed labels (i.e. force them to align)
        """
        all_labels: list[QLabel] = self.findChildren(QLabel)
        max_width = 0.0
        # HACK ignore non-field labels by internal knowledge (that all field-labels end with a colon ":")
        re_field_label = re.compile(r":\s*$")
        for lbl in all_labels:
            if not lbl.isVisible() or not re_field_label.search(lbl.text()):
                continue
            max_width = max(max_width, lbl.width())
        if max_width <= 0:
            return
        for lbl in all_labels:
            if not lbl.isVisible() or not re_field_label.search(lbl.text()):
                continue
            lbl.setMinimumWidth(max_width)

    def _createDlgCtrls(self):
        buttons = QDialogButtonBox()
        buttons.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Reset
            | QDialogButtonBox.StandardButton.Save
            # TODO?: opening is actually quite complicated (or un-intuitive) since we are storing config-changes in
            #        difference to a loaded config-file: if we change the config-file, the previous differences may
            #        not apply anymore... but it may be even more confusing, when on the next start the default
            #        config-file is loaded and the config-changes from "custom-loaded" config-file are applied to it...
            #        maybe it would be best, to only allow loading different config-file only via
            #        command-line (as it is now), that would probably less confusing for users when they expect certain
            # | QDialogButtonBox.StandardButton.Open FIXME maybe only via command-line?
        )
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.discardChangesAndReject)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(
            self.resetConfig
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(
            self.saveConfigToFile
        )
        return buttons

    def _createPortInput(self) -> Tuple[QSpinBox, QLabel, QVBoxLayout]:
        iptAvatarPort = QSpinBox()
        iptAvatarPort.setRange(0, 65535)
        port_config_path = CONFIG_ITEMS["avatar_ws_port"].config_path
        port_val, port_field, port_sub_config = get_current_value_and_config_path_for(
            self.config, port_config_path
        )
        if not isinstance(port_val, int) or port_val < 0:
            port_val = 0
        iptAvatarPort.setValue(port_val)
        # add info-label & also use it for invalid-values feedback (via self.setConfigValueAndValidation(), see below):
        infoLabel = QLabel(
            "(local port for converting avatar images with the browser renderer)"
        )

        iptAvatarPort.valueChanged.connect(
            partial(
                self.setConfigValueAndValidation,
                infoLabel,
                CONFIG_ITEMS["avatar_ws_port"].is_valid_value,
                port_config_path,
                port_sub_config,
                port_field,
            )
        )

        layout = QVBoxLayout()
        layout.addWidget(iptAvatarPort)
        layout.addWidget(infoLabel)
        return iptAvatarPort, infoLabel, layout

    def _createSliderFor(
        self,
        config_item: ConfigurationItem,
        min: int,
        max: int,
        step: int,
        additional_config_path: Optional[list[str]] = None,
    ) -> Tuple[QSlider, QHBoxLayout]:

        # slider for quick-change
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min, max)
        slider.setSingleStep(step)
        curr_val, field_name, sub_config = get_current_value_and_config_path_for(
            self.config, config_item.config_path
        )

        # "label" & number-input-field:
        spin = QSpinBox()
        spin.setRange(min, max)
        spin.setSingleStep(step)
        spin.setMinimumWidth(60)

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)

        slider.valueChanged.connect(
            partial(
                self.setConfigValueAndValidation,
                slider,
                config_item.is_valid_value,
                config_item.config_path,
                sub_config,
                field_name,
            )
        )

        if additional_config_path:
            _, add_field_name, add_sub_config = get_current_value_and_config_path_for(
                self.config, additional_config_path
            )
            slider.valueChanged.connect(
                partial(
                    self.setConfigValueAndValidation,
                    slider,
                    config_item.is_valid_value,  # FIXME should this be set to None? ... generally the validation should be the same, but there may be some unforseen cases where that is not the case...
                    additional_config_path,
                    add_sub_config,
                    add_field_name,
                )
            )

        # ensure valid (int) value
        if curr_val < min:
            curr_val = min
        elif curr_val > max:
            curr_val = max
        elif not isinstance(curr_val, int):
            curr_val = min

        # set (valid) value (will apply it to configuration, if it was changed above):
        slider.setValue(curr_val)

        layout = QHBoxLayout()
        layout.addWidget(spin)
        layout.addWidget(slider)

        return slider, layout

    def _createComboBoxFor(
        self, config_item: ConfigurationItem, update_values: bool = True
    ) -> QComboBox:
        config_values = (
            config_item.get_latest(self.config)
            if update_values
            else config_item.get(self.config)
        )
        combobox = QComboBox()
        if isinstance(config_values, list):
            return self._initTextComboBox(
                combobox, config_values, config_path=config_item.config_path
            )
        else:
            return self._initDataComboBox(
                combobox, config_values, config_path=config_item.config_path
            )

    def _initTextComboBox(
        self, combobox: QComboBox, text_items: list[str], config_path: list[str]
    ) -> QComboBox:
        """
        HELPER initialize a combo-box with simple text-items:
        on selection the corresponding text-value will be applied to the `config_path`.

        :param combobox: the combo-box to initialize
        :param text_items: the text-items; when selected their value will be applied to the config
        :param config_path: the path to the configuration value that will be updated
        :returns: the initialized combo-box
        """
        combobox.addItems(text_items)
        current_value, field_name, sub_config = get_current_value_and_config_path_for(
            self.config, config_path
        )
        # set current_value as selected index:
        try:
            idx = text_items.index(current_value)
            combobox.setCurrentIndex(idx)
        except:
            combobox.insertItem(0, NO_SELECTION)
            combobox.model().item(0).setEnabled(
                0
            )  # make NO_SELECTION entry non-selectable by users
            combobox.setCurrentIndex(0)
            self._createAndSetNoSelectionStyle(combobox)
        combobox.currentTextChanged.connect(
            partial(self.setConfigValue, combobox, config_path, sub_config, field_name)
        )
        return combobox

    def _initDataComboBox(
        self, combobox: QComboBox, item_data: dict[str, any], config_path: list[str]
    ) -> QComboBox:
        """
        HELPER initialize combo-box with data-items:
        the `key` will be used as label in the combo-box, and the `value` will be applied  to the `config_path`, when
        the corresponding item is selected.

        :param combobox: the combo-box to initialize
        :param item_data: the data-items, where the `key` is used as label and the `value` is applied to the config upon selection
        :param config_path: the path to the configuration value that will be updated
        :returns: the initialized combo-box
        """
        current_value, field_name, sub_config = get_current_value_and_config_path_for(
            self.config, config_path
        )
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
            combobox.model().item(0).setEnabled(
                0
            )  # make NO_SELECTION entry non-selectable by users
            combobox.setCurrentIndex(0)
            self._createAndSetNoSelectionStyle(combobox)
        combobox.currentIndexChanged.connect(
            partial(
                self.setConfigValueByData, combobox, config_path, sub_config, field_name
            )
        )
        return combobox

    def _createCheckBoxFor(
        self, config_item: ConfigurationItem, update_values: bool = True
    ) -> QCheckBox:
        # config_values = config_item.get_latest() if update_values else config_item.get()
        config_path = config_item.config_path
        checkbox = QCheckBox()
        current_value, field_name, sub_config = get_current_value_and_config_path_for(
            self.config, config_path
        )
        # if not None, set current_value as check/unchecked state:
        try:
            if current_value is not None:
                checkbox.setCheckState(
                    Qt.CheckState.Checked if current_value else Qt.CheckState.Unchecked
                )
        except:
            pass
        checkbox.stateChanged.connect(
            partial(self.setConfigValue, checkbox, config_path, sub_config, field_name)
        )
        return checkbox

    def _createVideoInputDeviceWidget(
        self,
    ) -> Tuple[QHBoxLayout, QComboBox, QPushButton]:
        """
        HELPER for creating widget to select video-input-device:
        detecting video-input-devices takes quite some time, so use a button to let users manually start
        search for available video input devices

        :returns: a tuple `(layout, combobox, button)` where the `layout` contains the combo-box and
                  the button (for starting the device search), the `combobox` is the selection-widget for the
                  video input device, and `button` is the button for starting the detection of available video
                  input devices
        """
        # NOTE: detecting video-input-devices takes quite some time, so use a button to let users manually start
        #       search for available video input devices

        # TODO should find better solution for generating video device list, that also includes capabilities,
        #      i.e. other than trying out opencv2 indices ...

        cbbVideoIn = QComboBox()
        cfgVideoIn = CONFIG_ITEMS["video_input_devices"]
        if not cfgVideoIn.config_values:
            # if search for video devices was not done yet: just show disabled selection with current config-value
            val, _, _ = get_current_value_and_config_path_for(
                self.config, cfgVideoIn.config_path
            )
            cbbVideoIn.addItem(str(val), NO_SELECTION)
            cbbVideoIn.setEnabled(False)
        else:
            # if we already have search results, initialized combo-box with it
            # (without refreshing/getting latest results)
            self._initDataComboBox(
                cbbVideoIn, cfgVideoIn.get(self.config), cfgVideoIn.config_path
            )

        videoInLayout = QHBoxLayout()
        videoInLayout.addWidget(cbbVideoIn)

        def detect_video_input_devices():
            """HELPER: search for video-input-devices & update combo-box with results"""
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                cbbVideoIn.setEnabled(False)
                # NOTE this will disconnect ALL signals:
                #      currently there is only the `currentIndexChanged`, that gets re-connected in
                #       `_initDataComboBox()`, but if there were other signals on `cbbVideoIn` they would need to get
                #       reconnected manually!
                #      -> if one of these two assertions raise an exception, that would indicate that there are other
                #         signals connected and need to be (manually) reconnected after the combo-box is re-initialized!
                try:
                    assert cbbVideoIn.receivers(cbbVideoIn.currentIndexChanged) <= 1
                    assert cbbVideoIn.receivers(cbbVideoIn.currentTextChanged) <= 1
                except AssertionError as exc:
                    _logger.error(
                        "outdated implementation for Video-Input-Device selection (ComboBox), "
                        "see comments in code!",
                        exc_info=exc,
                        stack_info=True,
                    )
                    QMessageBox.warning(
                        self,
                        "Outdated Implementation: Please contact Developer",
                        "Please contact developers, and include the logging output.",
                    )
                cbbVideoIn.disconnect()
                cbbVideoIn.clear()
                self._initDataComboBox(
                    cbbVideoIn,
                    CONFIG_ITEMS["video_input_devices"].get_latest(self.config),
                    CONFIG_ITEMS["video_input_devices"].config_path,
                )
                cbbVideoIn.setEnabled(True)
            finally:
                QApplication.restoreOverrideCursor()

        # add button that lets users initiate the search
        btnDetectVideoIn = QPushButton("Detect")
        btnDetectVideoIn.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        )
        usage_info = "detect video input devices\nWARNING: will take some time!"
        # btnDetectVideoIn.setStatusTip(usage_info)
        btnDetectVideoIn.setToolTip(usage_info)
        btnDetectVideoIn.clicked.connect(detect_video_input_devices)

        videoInLayout.addWidget(btnDetectVideoIn)

        return videoInLayout, cbbVideoIn, btnDetectVideoIn

    def _createAndSetNoSelectionStyle(self, combobox: QComboBox):
        """
        HELPER: style NO_SELECTION as "invalid" entry with red text

        Adds the styling for the items & set the combo-box to show the currently selected value
        (which should be the NO_SELECTION) as "invalid".

        IMPORTANT: the NO_SELECTION entry must be at index/row 0, and it must be selected!
        """
        # styling for NO_SELECTION:
        # since the NO_SELECTION entry is selected: do style the combo-box text as red
        self._setStyleToInvalidSelection(combobox)
        # next style the items in the dropdown list:
        model = combobox.model()
        # NO_SELECTION item text to red
        model.setData(model.index(0, 0), QColor("red"), Qt.ItemDataRole.ForegroundRole)
        # set all other items' text to default brush
        # (since combobox text is currently set to red, they would also be rendered red
        #  if they are not specifically set)
        defaultBrush = QPalette().brush(
            QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText
        )
        for i in range(1, model.rowCount()):
            model.setData(
                model.index(i, 0), defaultBrush, Qt.ItemDataRole.ForegroundRole
            )

    def _setStyleToInvalidSelection(self, widget: QWidget):
        """
        HELPER style the widget to indicate that it has an invalid value

        NOTE: use `_resetStyleToValidSelection()` to reset the style, when the selection becomes valid
        """
        widget.setStyleSheet(STYLE_INVALID_SELECTION)

    def _resetStyleToValidSelection(self, widget: QWidget):
        if widget.styleSheet() == STYLE_INVALID_SELECTION:
            defaultBrush = QPalette().brush(
                QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText
            )
            widget.setStyleSheet("color: " + str(defaultBrush.color().name()))

    def setConfigValue(
        self,
        widget: QComboBox | QCheckBox,
        config_path: list[str],
        sub_config: dict,
        field: str,
        value: any,
    ):
        is_combo = isinstance(widget, QComboBox)
        is_check = isinstance(widget, QCheckBox)
        if is_check:
            # for check-box: convert checked/unchecked state to True/False:
            value = True if widget.checkState() == Qt.CheckState.Checked else False
        elif is_combo and widget.currentIndex() == -1:
            # NOTE: index -1 indicates that no item in the combo-box is selected
            value = NO_SELECTION
        _logger.debug("set config %s: %s -> %s", sub_config, field, value)
        if value != NO_SELECTION:
            sub_config[field] = value
            self.changed_config[CONFIG_PATH_SEP.join(config_path)] = value
            if is_combo:
                self._resetStyleToValidSelection(widget)
        else:
            _logger.warning(
                "selected NO_SELECTION for %s: ignoring change (keeping old value: %s)!",
                field,
                sub_config.get(field),
            )
            if is_combo:
                self._setStyleToInvalidSelection(widget)

    def setConfigValueByData(
        self,
        widget: QComboBox,
        config_path: list[str],
        sub_config: dict,
        field: str,
        idx: int,
    ):
        # NOTE: index -1 indicates that no item in the combo-box is selected
        data = widget.itemData(idx) if idx > -1 else NO_SELECTION
        _logger.debug(
            "set config by index %s: %s -> ComboBox[%s] = [%s]",
            sub_config,
            field,
            idx,
            data,
        )
        if data != NO_SELECTION:
            sub_config[field] = data
            self.changed_config[CONFIG_PATH_SEP.join(config_path)] = data
            self._resetStyleToValidSelection(widget)
        else:
            _logger.warning(
                "selected NO_SELECTION for %s: ignoring change (keeping old value: %s)!",
                field,
                sub_config.get(field),
            )
            self._setStyleToInvalidSelection(widget)

    def setConfigValueAndValidation(
        self,
        widget: QSpinBox | QLabel,
        validation_func: Optional[Callable[[any, Optional[dict]], bool]],
        config_path: list[str],
        sub_config: dict,
        field: str,
        value: any,
    ):
        _logger.debug("set config %s: %s -> %s", sub_config, field, value)
        sub_config[field] = value
        self.changed_config[CONFIG_PATH_SEP.join(config_path)] = value

        is_invalid = not validation_func(value, None) if validation_func else False
        if not is_invalid:
            self._resetStyleToValidSelection(widget)
        else:
            _logger.warning(
                "selected INVALID value for %s (did set invalid value, user needs to correct this)!",
                field,
            )
            self._setStyleToInvalidSelection(widget)

    def saveConfigToFile(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "YAML Files(*.yaml);;All Files(*)"
        )
        if filePath:
            yaml.dump(self.config, open(filePath, "w"))

    def resetConfig(self):
        """
        reset config-changes and indicate to dialog owner that it should also reset the configuration by setting
        `isResetConfig` to `True`

        (will close the configuration dialog)
        """
        # signal to dialog owner, that it should discard current config / reload the default settings
        self.isResetConfig = True
        # reset config-changes
        # NOTE: also discard stored config changes, so that only the config loaded from initial file remains!
        self._discardConfigChanges(reset_stored_changes=True)
        # close the dialog (CANCEL)
        self.reject()

    def discardChangesAndReject(self):
        # reset config-changes:
        self._discardConfigChanges(reset_stored_changes=False)
        # close the dialog (CANCEL
        self.reject()

    def validateAndAccept(self):
        has_invalid = False
        errors = {}

        # NOTE: iterate over ALL children of QFormLayout (instead of only ComboBoxes themselves) in order to access
        #       preceding labels for ComboBoxes, for creating details in error-messages in case of invalid values
        # boxes: list[QComboBox] = self.findChildren(QComboBox)

        formLayouts: list[QFormLayout] = self.findChildren(
            QFormLayout
        )  # self.layout().itemAt(0)
        for formLayout in formLayouts:
            for i in range(formLayout.count()):
                widget = formLayout.itemAt(i).widget()
                if isinstance(widget, QComboBox):

                    # TODO use ConfigurationItem.can_ignore_validation()
                    #      instead of duplicating logic via widget.isEnabled()
                    #      ... or use UnitTesting to make sure both are in sync
                    if not widget.isEnabled():
                        # ignore, if combo-box is disabled:
                        continue
                    # validate selection in combo-box:
                    #   must not have NO_SELECTION as current selection
                    sel = widget.currentText()
                    if sel == NO_SELECTION:
                        has_invalid = True
                        did_add_error = False
                        if i > 0:
                            # get text from combo-box's (preceding) label
                            label = formLayout.itemAt(i - 1).widget()
                            if label and isinstance(label, QLabel):
                                did_add_error = True
                                label_str = label.text()
                                # add colon, if not present (for error message formatting)
                                if ":" not in label_str:
                                    label_str += ":"
                                errors[label_str] = sel
                        if not did_add_error:
                            errors["unknown"] = sel

        # validate non-combo-box values, if necessary
        # FIXME should move GUI labels to ConfigurationItems and only use this validation
        #       (instead of "validating" the GUI widgets)
        if not has_invalid:
            errs = validate_config_values(self.config)
            if errs:
                has_invalid = True
                for err in errs:
                    errors[err] = (
                        ""  # NOTE the error-entry are already complete, so just set empty string as value
                    )

        if not has_invalid:
            self.accept()
        else:
            details = ""
            for msg in errors:
                details += "\n * {} {}".format(msg, errors[msg])
            QMessageBox.question(
                self,
                "Invalid Value",
                "Please select a different value(s) for " + details,
                QMessageBox.StandardButton.Ok,
            )
