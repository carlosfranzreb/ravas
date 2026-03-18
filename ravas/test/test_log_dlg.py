import pytest
from PyQt6.QtWidgets import QApplication
import logging

from stream_processing.gui.log_dlg import LogDialog, QTextEditLogger

@pytest.fixture
def log_dialog(qtbot):
    """
    Fixture to create a LogDialog instance for testing.
    Adds the dialog to qtbot for proper cleanup.
    """
    dlg = LogDialog(parent=None)
    qtbot.addWidget(dlg)
    dlg.show()
    return dlg

def test_qtexteditlogger_lifecycle(qtbot):
    """
    Test the lifecycle of QTextEditLogger, including:
    - Initialization and read-only state
    - Adding messages via add()
    - Logging via emit()
    - CopyAll functionality to clipboard
    - Reset functionality
    """
    handler = QTextEditLogger(parent=None)
    qtbot.addWidget(handler.widget)
    
    # Initialization
    assert handler.widget.isReadOnly(), "Widget should be read-only"
    assert handler.widget.toPlainText() == "", "Widget should start empty"

    # ADD function
    direct_msg = "Direct Message"
    handler.add(direct_msg)
    assert direct_msg in handler.widget.toPlainText()

    # EMIT function
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="Error Occurred",
        args=None,
        exc_info=None
    )
    handler.emit(record)
    current_text = handler.widget.toPlainText()
    assert "ERROR: Error Occurred" in current_text

    # COPYALL function
    clipboard = QApplication.clipboard()
    clipboard.setText("Initial Clipboard Content")
    handler.copyAll()
    clip_text = clipboard.text()
    assert "Direct Message" in clip_text
    assert "ERROR: Error Occurred" in clip_text

    # Ensure no text is selected after copyAll
    cursor = handler.widget.textCursor()
    assert not cursor.hasSelection()
    assert cursor.atEnd()

    # RESET function
    handler.reset()
    assert handler.widget.toPlainText() == "", "Widget should be empty after reset"

def test_log_dialog_ui_connections(log_dialog, qtbot):
    """
    Test LogDialog UI elements and their functionality:
    - Window title
    - Toolbar actions
    - Reset, Copy All, and Close actions
    """
    dlg = log_dialog
    assert dlg.windowTitle() == "Logging"

    reset_action = next((a for a in dlg.actions() if "Reset" in a.text()), None)
    copy_action = next((a for a in dlg.actions() if "Copy All" in a.text()), None)
    close_action = next((a for a in dlg.actions() if "lose" in a.text()), None)

    assert reset_action is not None
    assert copy_action is not None
    assert close_action is not None

    # Test Reset action
    dlg.logWidget.add("Hello World")
    reset_action.trigger()
    assert dlg.logWidget.widget.toPlainText() == ""

    # Test Copy All action
    dlg.logWidget.add("Copy This")
    clipboard = QApplication.clipboard()
    clipboard.clear()
    copy_action.trigger()
    assert "Copy This" in clipboard.text()

    # Test Close action
    dlg.show()
    close_action.trigger()
    assert not dlg.isVisible()