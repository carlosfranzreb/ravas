import logging
from typing import Optional, Callable

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog
)


_logger = logging.getLogger('gui.settings_helper')


class RestoreSettingItem:
    def __init__(self, field: str, apply_func: Callable[[any], any], convert_func: Optional[Callable[[any], any]] = None):
        self.field = field
        self.apply_func = apply_func
        self.covert_func = convert_func


class StoreSettingItem:
    def __init__(self, field: str, get_func: Callable[[], any], convert_func: Optional[Callable[[any], any]] = None):
        self.field = field
        self.get_func = get_func
        self.covert_func = convert_func


class RestorableDialog(QDialog):

    def getSettingsPath(self) -> str:
        """
        must override in inherited classes to return path-like string that identifies the dialog (settings), e.g.
        `my_dialog` or `my_other_dialog/instance1`
        """
        raise NotImplemented('must implement getSettingsPath() to return path-like string that identifies dialog')

    def getRestoreSettingsItems(self) -> list[RestoreSettingItem]:
        """ override in inherited classes to add/change restored settings """
        settings_path = self.getSettingsPath()
        return [
            RestoreSettingItem(settings_path + '/size', self.resize),
            RestoreSettingItem(settings_path + '/position', self.move),
            RestoreSettingItem(settings_path + '/windowState', self.setWindowState, checkWindowState),
        ]

    def getStoreSettingsItems(self) -> list[StoreSettingItem]:
        """ override in inherited classes to add/change restored settings """
        settings_path = self.getSettingsPath()
        return [
            StoreSettingItem(settings_path + '/size', self.size),
            StoreSettingItem(settings_path + '/position', self.pos),
            StoreSettingItem(settings_path + '/windowState', self.windowState, checkWindowState),
        ]

    def getSettings(self) -> QSettings:
        """ should override in inherited classes, e.g. store in field and return field here """
        return QSettings()

    # override QDialog.showEvent (startup):
    def showEvent(self, evt):
        items = self.getRestoreSettingsItems()
        settings = self.getSettings()
        for c in items:
            applySetting(settings, c.field, c.apply_func, c.covert_func)
        super().showEvent(evt)

    # override QDialog.done (shutdown):
    def done(self, code):
        items = self.getStoreSettingsItems()
        settings = self.getSettings()
        for c in items:
            storeSetting(settings, c.field, c.get_func, c.covert_func)
        super().done(code)


def applySetting(settings: QSettings, field: str, apply_func: Callable[[any], any], convert_func: Optional[Callable[[any], any]] = None):
    val = settings.value(field, None)
    if val is not None:
        if convert_func:
            val = convert_func(val)
            if val is None:
                # _logger.info('did not restore %s (FILTERED): %s', field, val)
                return
        # _logger.info('restore %s: %s', field, val)
        apply_func(val)


def storeSetting(settings: QSettings, field: str, get_func: Callable[[], any], convert_func: Optional[Callable[[any], any]] = None):
    val = get_func()
    if val is not None:
        if convert_func:
            val = convert_func(val)
            if val is None:
                # _logger.info('did not store %s setting (FILTERED): %s', field, val)
                return
        # _logger.info('store setting %s: %s', field, val)
        settings.setValue(field, val)


def checkWindowState(state: Qt.WindowState) -> Qt.WindowState | None:
    if state == Qt.WindowState.WindowMaximized or state == Qt.WindowState.WindowFullScreen:
        return state
    return None
