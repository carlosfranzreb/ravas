import logging

from PyQt6.QtGui import QAction, QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit, QVBoxLayout, QToolBar

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

    def reset(self):
        self.widget.clear()

    def copyAll(self):
        sels = self.widget.extraSelections()
        self.widget.selectAll()
        self.widget.copy()

        # HACK deselect text (and move to end position):
        cursor = self.widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.widget.setTextCursor(cursor)

        self.widget.setExtraSelections(sels)


class LogDialog(RestorableDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setWindowTitle("Logging")

        logTextBox = QTextEditLogger(self)
        logTextBox.setFormatter(
            logging.Formatter(
                "%(asctime)s %(processName)-10s %(process)-8d %(name)s %(levelname)-8s %(message)s"
            )
        )
        logging.getLogger().addHandler(logTextBox)
        self.logWidget = logTextBox

        actClose = QAction("C&lose", self)
        actClose.setShortcut(
            "Ctrl+L"
        )  # NOTE use CTRL-L, same as for toggling log-window in MainWindow
        actClose.setToolTip("Close Logging Window")
        actClose.triggered.connect(self.close)
        self.addAction(actClose)

        actReset = QAction("&Reset", self)
        actReset.setShortcut("Ctrl+R")
        actReset.setToolTip("Reset Logging Output")
        actReset.triggered.connect(logTextBox.reset)
        self.addAction(actReset)

        actCopyAll = QAction("&Copy All", self)
        actCopyAll.setShortcut("Ctrl+Shift+C")
        actCopyAll.setToolTip("Copy All Logging Output to Clipboard")
        actCopyAll.triggered.connect(logTextBox.copyAll)
        self.addAction(actCopyAll)

        tools = QToolBar()
        tools.addAction(actReset)
        tools.addAction(actCopyAll)
        tools.addAction(actClose)

        layout = QVBoxLayout()
        layout.addWidget(tools)
        layout.addWidget(logTextBox.widget)
        self.setLayout(layout)

    # override RestorableDialog.getSettings():
    def getSettingsPath(self) -> str:
        return "log_dlg"
