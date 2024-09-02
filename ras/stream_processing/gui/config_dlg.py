import logging
from copy import deepcopy
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QDialogButtonBox, QFormLayout, QVBoxLayout, QComboBox, QMessageBox, QLabel, QGroupBox, \
    QSizePolicy, QWidget

from .config_items import CONFIG_ITEMS, NO_SELECTION, ConfigurationItem
from .settings_helper import RestorableDialog


_logger = logging.getLogger('gui.config_dlg')


STYLE_INVALID_SELECTION = 'color: red'
""" widget style to indicate an invalid selection/configuration """


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

        self.config = deepcopy(config)

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

    def showEvent(self, evt):
        super().showEvent(evt)
        self._forceAlignRowLabels()

    def _forceAlignRowLabels(self):
        """ HELPER set all label's minimum width to the maximum of the currently displayed labels (i.e. force them to align)"""
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
        )
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.reject)
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
        current_value, field_name, sub_config = self._get_current_value_and_config_path_for(config_path)
        # set current_value as selected index:
        try:
            idx = text_items.index(current_value)
            combobox.setCurrentIndex(idx)
        except:
            combobox.insertItem(0, NO_SELECTION)
            combobox.setCurrentIndex(0)
            self._createAndSetNoSelectionStyle(combobox)
        combobox.currentTextChanged.connect(partial(self.setConfigValue, combobox, sub_config, field_name))
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
            self._createAndSetNoSelectionStyle(combobox)
        combobox.currentIndexChanged.connect(partial(self.setConfigValueByData, combobox, sub_config, field_name))
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

    def setConfigValue(self, widget: QComboBox, sub_config: dict, field: str, value: any):
        _logger.debug('set config %s: %s -> %s', sub_config, field, value)
        if value != NO_SELECTION:
            sub_config[field] = value
            self._resetStyleToValidSelection(widget)
        else:
            _logger.warning('selected NO_SELECTION for %s: reset config value to None!', field)
            sub_config[field] = None
            self._setStyleToInvalidSelection(widget)

    def setConfigValueByData(self, widget: QComboBox, sub_config: dict, field: str, idx: int):
        data = widget.itemData(idx)
        _logger.debug('set config by index %s: %s -> ComboBox[%s] = [%s]', sub_config, field, idx, data)
        if data != NO_SELECTION:
            sub_config[field] = data
            self._resetStyleToValidSelection(widget)
        else:
            _logger.warning('selected NO_SELECTION for %s: reset config value to None!', field)
            sub_config[field] = None
            self._setStyleToInvalidSelection(widget)

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
