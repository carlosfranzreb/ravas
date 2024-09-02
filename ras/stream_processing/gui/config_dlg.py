import json
import logging
from copy import deepcopy
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QDialogButtonBox, QFormLayout, QVBoxLayout, QComboBox, QMessageBox, QLabel, QGroupBox, \
    QSizePolicy, QWidget

from .config_items import CONFIG_ITEMS, NO_SELECTION, ConfigurationItem
from .config_utils import get_current_value_and_config_path_for
from .settings_helper import RestorableDialog, RestoreSettingItem, StoreSettingItem, applySetting


_logger = logging.getLogger('gui.config_dlg')


STYLE_INVALID_SELECTION = 'color: red'
""" widget style to indicate an invalid selection/configuration """

CONFIG_PATH_SEP = '/'
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
        self.changed_config: dict
        """ 
        flattened dictionary containing changed config values (changed w.r.t. to the original config-file)
        (stored/restored from settings; the keys are "flattened" using `CONFIG_PATH_SEP` as separator)
        """

        # do restore config-changes from settings and apply them:
        #   (load them "flattened" into self.changed_config, and apply them to self.config)
        self._restoreAndApplyConfigChangeSettings()

        dialogLayout = QVBoxLayout()

        # ############ INPUT ################
        inputForm = QFormLayout()

        # cbbAudioIn = self._createAudioDeviceWidget(audio_input=True)
        cbbAudioIn = self._createWidgetFor(CONFIG_ITEMS['audio_input_devices'])
        inputForm.addRow("Audio Input:", cbbAudioIn)

        # FIXME DISABLED takes too long, find better solution for generating video device list,
        #                other than trying out opencv2 indicies ... or maybe show fast solution first and give users
        #                the option to start a time consuming search, if they want to
        # cbbVideoIn = self._createVideoDeviceWidget()
        cbbVideoIn = QComboBox()  # FIXME dummy
        cbbVideoIn.addItem('<disabled>')  # FIXME dummy
        cbbVideoIn.setEnabled(False)  # FIXME dummy
        inputForm.addRow("Video Input:", cbbVideoIn)

        inputGroup = self._makeGroupBox("Input", inputForm)
        dialogLayout.addWidget(inputGroup)

        # ############ OUTPUT ################
        outputForm = QFormLayout()

        # cbbAudioOut = self._createAudioDeviceWidget(audio_input=False)
        cbbAudioOut = self._createWidgetFor(CONFIG_ITEMS['audio_output_devices'])
        outputForm.addRow("Audio Output:", cbbAudioOut)

        outputGroup = self._makeGroupBox("Output", outputForm)
        dialogLayout.addWidget(outputGroup)

        # ############ CONVERTER AUDIO ################

        convertAudioForm = QFormLayout()

        cbbAudioVoice = self._createWidgetFor(CONFIG_ITEMS['audio_voices'])
        convertAudioForm.addRow("Audio Voice:", cbbAudioVoice)

        convertAudioGroup = self._makeGroupBox("Convert Audio", convertAudioForm)
        dialogLayout.addWidget(convertAudioGroup)

        # ############ CONVERTER VIDEO ################
        convertVideoForm = QFormLayout()

        # cbbVideoConverter = self._createVideoConverterWidget()
        cbbVideoConverter = self._createWidgetFor(CONFIG_ITEMS['video_converters'])
        convertVideoForm.addRow("Video Converter:", cbbVideoConverter)

        # cbbAvatar = self._createAvatarWidget()
        cbbAvatar = self._createWidgetFor(CONFIG_ITEMS['video_avatars'])
        convertVideoForm.addRow("Avatar:", cbbAvatar)

        # MOD cbbVideoConverter: enable/disable cbbAvatar when converter "Avatar" is selected/deselected
        def _updateAvatarEnabled(selected_video_converter: str):
            cbbAvatar.setEnabled(selected_video_converter == 'Avatar')
        _updateAvatarEnabled(cbbVideoConverter.currentText())
        cbbVideoConverter.currentTextChanged.connect(_updateAvatarEnabled)

        convertVideoGroup = self._makeGroupBox("Convert Video", convertVideoForm)
        dialogLayout.addWidget(convertVideoGroup)

        # ############ LOG SETTINGS ################
        loggingForm = QFormLayout()

        # cbbMainLogging = self._createLogLevelWidget(for_gui=False)
        cbbMainLogging = self._createWidgetFor(CONFIG_ITEMS['log_levels'])
        loggingForm.addRow("Logging:", cbbMainLogging)

        # cbbGuiLogging = self._createLogLevelWidget(for_gui=True)
        cbbGuiLogging = self._createWidgetFor(CONFIG_ITEMS['gui_log_levels'])
        loggingForm.addRow("Logging (GUI):", cbbGuiLogging)

        loggingGroup = self._makeGroupBox("Logging", loggingForm)
        dialogLayout.addWidget(loggingGroup)

        # TODO add config widgets for:
        # use_audio: true
        #
        # use_video: true
        # output_window: true
        #
        # disable_console_logging   TODO
        #
        # ? [video]/[converter]
        # max_fps: 20
        # width: *video_width
        # height: *video_height

        buttons = self._createDlgCtrls()
        dialogLayout.addWidget(buttons)

        self.setLayout(dialogLayout)

    def _makeGroupBox(self, title: str, layout: QFormLayout):
        group = QGroupBox(title)
        group.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum))  # prevent vertical growth
        group.setLayout(layout)
        return group

    # override RestorableDialog.getSettings():
    def getSettingsPath(self) -> str:
        return 'config_dlg'

    def getStoreSettingsItems(self) -> list[StoreSettingItem]:
        # NOTE: only override getStoreSettingsItems(), and not getRestoreSettingsItems():
        #       the config-changes need to be restored before showing the dialog, within the constructor,
        #       so simply overriding getRestoreSettingsItems() would apply them too late
        #       -> see _restoreAndApplyConfigChangeSettings() for loading & applying the config-changes
        settings_path = self.getSettingsPath()
        settings = super().getStoreSettingsItems()
        settings.append(StoreSettingItem(settings_path + '/configChanges', self._storeConfigChanges))
        return settings

    def showEvent(self, evt):
        super().showEvent(evt)
        self._forceAlignRowLabels()

    def closeEvent(self, evt):
        # if close-button on window is used:
        # the configuration dialog was canceled
        # -> discard config-changes BEFORE handling close-event in parent-implementation
        #    (e.g. before storing to user settings9
        self._discardConfigChanges()
        super().closeEvent(evt)

    def _restoreAndApplyConfigChangeSettings(self):
        settings = self.getSettings()
        c = RestoreSettingItem(self.getSettingsPath() + '/configChanges', self._applyConfigChanges)
        applySetting(settings, c.field, c.apply_func, c.covert_func)

    def _applyConfigChanges(self, json_string: str):
        _logger.debug('applying config changes: %s', json_string)
        try:
            self.changed_config: dict = json.loads(json_string)
        except Exception as exc:
            _logger.error('failed to restore configuration changes: could not changes parse as JSON -> "%s"', json_string, exc_info=exc)
            self.changed_config = {}
        _logger.debug('applying config changes: %s', self.changed_config)
        for path_str, changed_value in self.changed_config.items():
            config_path = path_str.split(CONFIG_PATH_SEP)
            curr_val, field, sub_config = get_current_value_and_config_path_for(self.config, config_path)
            # _logger.debug('  applying change ', config_path, ' = ', curr_val, ' <- ', changed_value)
            sub_config[field] = changed_value

    def _storeConfigChanges(self) -> str:
        json_string = json.dumps(self.changed_config)
        _logger.debug('storing config changes: %s', json_string)
        return json_string

    def _discardConfigChanges(self):
        self.changed_config.clear()

    def _forceAlignRowLabels(self):
        """
        HELPER set all label's minimum width to the maximum of the currently displayed labels (i.e. force them to align)
        """
        all_labels: list[QLabel] = self.findChildren(QLabel)
        max_width = .0
        for lbl in all_labels:
            max_width = max(max_width, lbl.width())
        for lbl in all_labels:
            lbl.setMinimumWidth(max_width)

    def _createDlgCtrls(self):
        buttons = QDialogButtonBox()
        buttons.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Reset
        )
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.discardChangesAndReject)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.resetConfig)
        return buttons

    def _createWidgetFor(self, config_item: ConfigurationItem, update_values: bool = True):
        config_values = config_item.get_latest() if update_values else config_item.get()
        if isinstance(config_values, list):
            return self._createTextComboBox(config_values, config_path=config_item.config_path)
        else:
            return self._createDataComboBox(config_values, config_path=config_item.config_path)

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
        current_value, field_name, sub_config = get_current_value_and_config_path_for(self.config, config_path)
        # set current_value as selected index:
        try:
            idx = text_items.index(current_value)
            combobox.setCurrentIndex(idx)
        except:
            combobox.insertItem(0, NO_SELECTION)
            combobox.model().item(0).setEnabled(0)  # make NO_SELECTION entry non-selectable by users
            combobox.setCurrentIndex(0)
            self._createAndSetNoSelectionStyle(combobox)
        combobox.currentTextChanged.connect(partial(self.setConfigValue, combobox, config_path, sub_config, field_name))
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
        current_value, field_name, sub_config = get_current_value_and_config_path_for(self.config, config_path)
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
            combobox.model().item(0).setEnabled(0)  # make NO_SELECTION entry non-selectable by users
            combobox.setCurrentIndex(0)
            self._createAndSetNoSelectionStyle(combobox)
        combobox.currentIndexChanged.connect(partial(self.setConfigValueByData, combobox, config_path, sub_config, field_name))
        return combobox

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
        model.setData(model.index(0, 0), QColor('red'), Qt.ItemDataRole.ForegroundRole)
        # set all other items' text to default brush
        # (since combobox text is currently set to red, they would also be rendered red
        #  if they are not specifically set)
        defaultBrush = QPalette().brush(QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText)
        for i in range(1, model.rowCount()):
            model.setData(model.index(i, 0), defaultBrush, Qt.ItemDataRole.ForegroundRole)

    def _setStyleToInvalidSelection(self, widget: QWidget):
        """
        HELPER style the widget to indicate that it has an invalid value

        NOTE: use `_resetStyleToValidSelection()` to reset the style, when the selection becomes valid
        """
        widget.setStyleSheet(STYLE_INVALID_SELECTION)

    def _resetStyleToValidSelection(self, widget: QWidget):
        if widget.styleSheet() == STYLE_INVALID_SELECTION:
            defaultBrush = QPalette().brush(QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText)
            widget.setStyleSheet('color: ' + str(defaultBrush.color().name()))

    def setConfigValue(self, widget: QComboBox, config_path: list[str], sub_config: dict, field: str, value: any):
        _logger.debug('set config %s: %s -> %s', sub_config, field, value)
        if value != NO_SELECTION:
            sub_config[field] = value
            self.changed_config[CONFIG_PATH_SEP.join(config_path)] = value
            self._resetStyleToValidSelection(widget)
        else:
            _logger.warning('selected NO_SELECTION for %s: ignoring change (keeping old value: %s)!', field, sub_config.get(field))
            self._setStyleToInvalidSelection(widget)

    def setConfigValueByData(self, widget: QComboBox, config_path: list[str], sub_config: dict, field: str, idx: int):
        data = widget.itemData(idx)
        _logger.debug('set config by index %s: %s -> ComboBox[%s] = [%s]', sub_config, field, idx, data)
        if data != NO_SELECTION:
            sub_config[field] = data
            self.changed_config[CONFIG_PATH_SEP.join(config_path)] = data
            self._resetStyleToValidSelection(widget)
        else:
            _logger.warning('selected NO_SELECTION for %s: ignoring change (keeping old value: %s)!', field, sub_config.get(field))
            self._setStyleToInvalidSelection(widget)

    def resetConfig(self):
        """
        reset config-changes and indicate to dialog owner that it should also reset the configuration by setting
        `isResetConfig` to `True`

        (will close the configuration dialog)
        """
        # signal to dialog owner, that it should discard current config / reload the default settings
        self.isResetConfig = True
        # reset config-changes:
        self._discardConfigChanges()
        # close the dialog (CANCEL)
        self.reject()

    def discardChangesAndReject(self):
        # reset config-changes:
        self._discardConfigChanges()
        # close the dialog (CANCEL
        self.reject()

    def validateAndAccept(self):
        has_invalid = False
        errors = {}

        # NOTE: iterate over ALL children of QFormLayout (instead of only ComboBoxes themselves) in order to access
        #       preceding labels for ComboBoxes, for creating details in error-messages in case of invalid values
        # boxes: list[QComboBox] = self.findChildren(QComboBox)

        formLayouts: list[QFormLayout] = self.findChildren(QFormLayout)  # self.layout().itemAt(0)
        for formLayout in formLayouts:
            for i in range(formLayout.count()):
                widget = formLayout.itemAt(i).widget()
                if isinstance(widget, QComboBox):
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
