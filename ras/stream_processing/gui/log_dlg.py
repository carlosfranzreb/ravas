import logging

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout

from .settings_helper import RestorableDialog


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)

    def add(self, message):
        # print('calling QTextEditLogger.add(..)', flush=True)
        self.widget.appendPlainText(message)


class LogDialog(RestorableDialog):
    def __init__(self, parent, gui_log_level='INFO'):
        super().__init__(parent=parent)
        self.setWindowTitle('Logging')

        actClose = QAction("Close", self)
        actClose.setShortcut("Ctrl+L")  # NOTE use CTRL-L, same as for toggling log-window in MainWindow
        actClose.setStatusTip("Close Logging Window")
        actClose.triggered.connect(self.close)
        self.addAction(actClose)

        logTextBox = QTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter(
            "%(asctime)s %(processName)-10s %(process)-8d %(name)s %(levelname)-8s %(message)s"
        ))
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(gui_log_level)
        self.logWidget = logTextBox

        layout = QVBoxLayout()
        layout.addWidget(logTextBox.widget)
        self.setLayout(layout)

    # override RestorableDialog.getSettings():
    def getSettingsPath(self) -> str:
        return 'log_dlg'
