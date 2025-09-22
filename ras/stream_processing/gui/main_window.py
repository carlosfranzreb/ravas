import logging
import time
import wave
import os
import subprocess

from copy import deepcopy
from typing import Optional

import yaml
from PyQt6.QtCore import QThread, QThreadPool, QSettings
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QToolBar,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QCheckBox,
    QToolButton,
    QMessageBox,
)
from torch import multiprocessing

from .config_dlg import ConfigDialog
from .config_utils import validate_config_values
from .gui_logging import init_gui_logging, LogWorker
from .log_dlg import LogDialog
from .settings_helper import storeSetting, checkWindowState, applySetting
from .task import Task
from ..streamer import AudioVideoStreamer


_logger = logging.getLogger("gui.main_window")


class MainWindow(QMainWindow):

    def __init__(self, config_path: str):
        super().__init__(parent=None)
        self.setWindowTitle("RAVAS")

        self._audioVideoStreamer: Optional[AudioVideoStreamer] = None
        self._config_path: str = config_path
        self._config: Optional[dict] = None

        self._log_worker: Optional[LogWorker] = None
        self._log_thread: Optional[QThread] = None
        self._is_log_stopping: bool = False

        self._threadpool = QThreadPool()

        self._createMain()
        self._createMenu()
        self._createToolBar()
        self._createStatusBar()

    def _createMain(self):
        main = QWidget(parent=self)
        layout = QHBoxLayout()

        actStart = QAction("&Start Streaming", self)
        actStart.setStatusTip("Start streaming Audio/Video")
        actStart.triggered.connect(self.startStreaming)
        self._actStart = actStart

        actStop = QAction("Sto&p Streaming", self)
        actStop.setStatusTip("Stop streaming Audio/Video")
        actStop.triggered.connect(self.stopStreaming)
        self._actStop = actStop

        actShowConfig = QAction("&Configuration", self)
        actShowConfig.setStatusTip("Open the Configuration Dialog")
        actShowConfig.triggered.connect(self.showConfigDialog)
        self._actShowConfig = actShowConfig

        actExit = QAction("&Quit", self)
        actExit.setShortcut("Ctrl+Q")
        actExit.setStatusTip("Quit the Application")
        actExit.triggered.connect(self.close)
        self._actExit = actExit

        btnStart = QToolButton()
        btnStart.setDefaultAction(self._actStart)
        layout.addWidget(btnStart)

        btnStop = QToolButton()
        btnStop.setDefaultAction(self._actStop)
        layout.addWidget(btnStop)

        btnDebug = QPushButton("Debug")
        btnDebug.setStatusTip("print some debug information to console")
        btnDebug.clicked.connect(self.__debug)
        layout.addWidget(btnDebug)

        self._logWindow = LogDialog(parent=self)
        self._logWindow.finished.connect(self.onLogWindowClosed)
        self._applyLogLevel()  # <- do set/update log-level for this process' root logger

        chkShowFrame = QCheckBox("Show Log")
        chkShowFrame.setShortcut("Ctrl+L")
        chkShowFrame.setStatusTip("Show or hide the Logging Window")
        chkShowFrame.toggled.connect(self.toggleLogWindow)
        layout.addWidget(chkShowFrame)
        self._chkShowFrame = chkShowFrame

        main.setLayout(layout)
        self.setCentralWidget(main)

        self._updateUiForStreaming(is_active=False)

        self._restoreSettings()

    def _createMenu(self):
        menu = self.menuBar().addMenu("&Menu")
        # menu.addAction("&Start Streaming", self.startStreaming)
        menu.addAction(self._actStart)
        menu.addAction(self._actStop)
        menu.addAction(self._actShowConfig)
        menu.addAction(self._actExit)

    def _createToolBar(self):
        tools = QToolBar()
        tools.addAction(self._actStart)
        tools.addAction(self._actStop)
        tools.addAction(self._actShowConfig)
        tools.addAction(self._actExit)
        self.addToolBar(tools)

    def _createStatusBar(self):
        status = QStatusBar()
        # status.showMessage("this is the status bar")
        self.setStatusBar(status)

    def _restoreSettings(self):
        settings = QSettings()
        applySetting(settings, "main_window/size", self.resize)
        applySetting(settings, "main_window/position", self.move)
        applySetting(
            settings, "main_window/windowState", self.setWindowState, checkWindowState
        )

        # restore config-changes:
        # when config-dialog is created it will restore config-changes from user-settings into configDlg.config
        configDlg = ConfigDialog(self, self.getConfig(as_copy=True))
        if configDlg.changed_config:
            self._config = configDlg.config

    def _applyLogLevel(self, config: dict = None):
        if not config:
            config = self.getConfig(as_copy=False)
        gui_log_level = config.get("gui_log_level")
        # NOTE must explicitly check here and cannot use config.get(.., DEFAULT) for fallback config["log_level"],
        #      since it may have explicitly been set to None in config:
        if not gui_log_level:
            gui_log_level = config["log_level"]
        logging.getLogger().setLevel(gui_log_level)

    def getConfig(self, as_copy: bool) -> dict:
        if not self._config:
            with open(self._config_path, "r") as f:
                self._config = yaml.safe_load(f)
        return deepcopy(self._config) if as_copy else self._config

    def _setConfigShowFrame(self, enable: bool):
        # TODO
        config = self.getConfig(as_copy=False)
        if config:
            config_video = config.get("video")
            if config_video:
                config_video["output_window"] = enable

    def setStatusText(self, text: str):
        self.statusBar().showMessage(text)

    def startStreaming(self):

        self.config_copy = self.getConfig(as_copy=True)
        config_problems = validate_config_values(self.config_copy)
        if config_problems:
            _logger.warning(
                "found unknown configuration values (may be invalid and cause errors):\n  * "
                + "\n  * ".join(config_problems)
            )
            details = (
                "The configuration has some unknown values which \nmay be invalid and may cause errors:\n * "
                + ("\n * ".join(config_problems))
                + "\n\nDo you want to continue anyway?"
            )
            result = QMessageBox.question(
                self,
                "Unknown Configuration: Continue?",
                details,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Abort,
                defaultButton=QMessageBox.StandardButton.Abort,
            )
            if result == QMessageBox.StandardButton.Abort:
                return

        self.setStatusText("Start streaming...")
        self._updateUiForStreaming(is_active=True)

        # start_streaming() would block the GUI for some time, so as task:
        task = Task(start_streaming, self.config_copy)
        task.signals.result.connect(self._setAudioVideoStreamer)
        task.signals.error.connect(self._handleAudioVideoStreamerError)
        self._threadpool.start(task)

    def _updateUiForStreaming(self, is_active: bool):
        self._actStart.setEnabled(not is_active)
        self._actStop.setEnabled(is_active)
        self._actShowConfig.setEnabled(not is_active)

    def _setAudioVideoStreamer(self, args):
        """
        HELPER for handling task result of `start_streaming()`
        :param args: result of `start_streaming()`, i.e. `tuple[AudioVideoStreamer, LogWorker, QThread]`
        """
        self._audioVideoStreamer = args[0]  # av_streamer
        self._log_worker = args[1]  # log_worker
        self._log_thread = args[2]  # log_thread

        # hook-up the log-widget to log-worker's emit signal:
        self._log_worker.emitLogLine.connect(self._logWindow.logWidget.add)

        self._audioVideoStreamer.start()
        self.__debug()  # FIXME DEBUG

    def _handleAudioVideoStreamerError(
        self, err_info
    ):  # : tuple[ExceptionClass, ExceptionInstance, traceback_string]
        """
        HELPER for handling task errors
        :param err_info: error information as `tuple[ExceptionClass, ExceptionInstance, traceback_string]`
        """
        err_name = (
            err_info[0].__name__
            if hasattr(err_info[0], "__name__")
            else str(err_info[0])
        )
        msg = "{} occurred: {}".format(err_name, err_info[1])
        logging.getLogger().error("%s, %s", msg, err_info[2])
        self.setStatusText(msg)

    def get_wav_duration(self, filename):
        with wave.open(filename, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration

    def merge_audio_video(self, log_dir: str) -> None:
        """
        Merge the audio and video files into a single file.

        Args:
        - log_dir: The directory where the audio and video files are stored.
        """
        audio_file = os.path.join(log_dir, "audio.wav")
        input_audio_file = os.path.join(log_dir, "input_audio.wav")
        video_file = os.path.join(
            log_dir, "video." + self.config_copy["video"]["store_format"]
        )
        output_file = os.path.join(log_dir, "merged.mp4")
        # Calculates the stretch factor and atempo to delay our audio
        audio_duration = self.get_wav_duration(audio_file)
        input_duration = self.get_wav_duration(input_audio_file)

        stretch_factor = input_duration / audio_duration
        atempo_value = 1 / stretch_factor
        with open(os.path.join(log_dir, "ffmpeg.log"), "a") as f:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    audio_file,
                    "-i",
                    video_file,
                    "-filter:a",
                    f"atempo={atempo_value:.6f}",  # adjust tempo because ours is faster by a bit
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    output_file,
                ],
                stdout=f,
                stderr=f,
            )

    def stopStreaming(self):
        print("stop streaming...")  # FIXME DEBUG
        self.setStatusText("Stopping streaming...")

        if self._audioVideoStreamer:
            self._audioVideoStreamer.stop()
            self._audioVideoStreamer = None
            time.sleep(1)

        if self._log_worker:
            print("  stopping logger...")  # FIXME DEBUG

            def on_log_finished():
                print("------ did receive finish signal!", flush=True)  # FIXME DEBUG
                # self._log_thread = None
                if self._log_thread and self._log_thread.isRunning():
                    self._log_thread.quit()
                self._log_worker = None
                self._log_thread = None

                self.setStatusText("Stopped streaming.")
                self._updateUiForStreaming(is_active=False)

                print("stopped streaming!", flush=True)  # FIXME DEBUG

            self._log_worker.finished.connect(on_log_finished)
            self._log_thread.requestInterruption()

            log_dir = self.config_copy["log_dir"]
            if (
                self.config_copy["audio"]["store"]
                and self.config_copy["video"]["store"]
            ):
                self.merge_audio_video(log_dir)

            # TESTING
            # start = time.time()
            # # is_finished = self._log_thread.wait(2000)
            #
            # is_finished = False
            # for i in range(20):
            #     QApplication.processEvents()
            #     time.sleep(0.1)
            #     if not self._log_thread.isRunning():
            #         print('log thread stopped during wait!', flush=True)
            #         is_finished = True
            #         break
            #
            # print('waited %s seconds (result %s) for log thread to shutdown' % (is_finished, time.time() - start,))
            # if self._log_thread.isRunning():
            #     print('\n############# log thread still running')
            # else:
            #     self._log_thread = None

            # [russa] do not use thread.quit(): need to use stop signal (i.e. stop gracefully),
            #         so that worker does stop the process it started:
            # self._log_thread.quit()
            # [russa] ... also leave the reference for thread, in order to prevent premature garbage-collection and
            #         stopping the thread while it is still shutting down gracefully:
            # self._log_thread = None

    def _shutdown(self):
        self.stopStreaming()
        self._mainProc = multiprocessing.current_process()  # FIXME TEST

    def toggleLogWindow(self, checked):
        # print('toggle log window ', checked)
        if checked:
            self._logWindow.show()
        else:
            self._logWindow.close()

    def onLogWindowClosed(self):
        self._chkShowFrame.setChecked(False)

    def showConfigDialog(self):
        config = self.getConfig(as_copy=True)
        configDlg = ConfigDialog(parent=self, config=config)
        result = configDlg.exec()
        print("closed config dialog -> ", result, configDlg)
        if result:
            print("applying config form config dialog -> ", result, configDlg)
            self._config = configDlg.config
            self._applyLogLevel(self._config)
        elif configDlg.isResetConfig:
            print("resetting config!")
            # setting _config to None will cause reload of config file:
            self._config = None
            self._applyLogLevel()

    def closeEvent(self, evt: QCloseEvent):
        print("main window close", evt)
        # evt.setAccepted(False) # <- would prevent closing window

        settings = QSettings()
        storeSetting(settings, "main_window/size", self.size)
        storeSetting(settings, "main_window/position", self.pos)
        storeSetting(
            settings, "main_window/windowState", self.windowState, checkWindowState
        )

        self._shutdown()

    def __debug(self):
        # FIXME DEBUG

        def print_proc(label: str, proc: multiprocessing.Process):
            print(label % (proc.name, proc.pid), flush=True)

        print_proc("MAIN ----> %s (pid %s)", multiprocessing.current_process())

        # if self._logListener:
        #     print_proc('  LOGGING ## %s (pid %s)', self._logListener)

        streamer = self._audioVideoStreamer
        if not streamer:
            print("   AUDIO VIDEO STREAMER not started!")
            return

        if hasattr(streamer, "audio_handler"):
            for key, p in streamer.audio_handler.procs.items():
                print_proc("  AUDIO ~~ {: <7} ~~  %s (pid %s)".format(key), p)
        else:
            print("  AUDIO: <disabled>")

        if hasattr(streamer, "video_handler"):
            for key, p in streamer.video_handler.procs.items():
                print_proc("  VIDEO ** {: <7} **  %s (pid %s)".format(key), p)
        else:
            print("  VIDEO: <disabled>")


def start_streaming(config: dict) -> tuple[AudioVideoStreamer, LogWorker, QThread]:
    """
    - Create a logging directory, and store the config there.
    - Create a logging file in the logging directory.
    - Start the audio-video streamer with the given config.

    Args:
    - config: The config for the demonstrator.
    """

    # print('running start_streaming()', config)

    # check if the config is valid (initial check taken from main.py)
    proc_size = config["audio"]["processing_size"]
    buffer_size = config["audio"]["record_buffersize"]
    assert proc_size > buffer_size, "Proc. size should be greater than buffer size"

    log_worker, thread = init_gui_logging(config)

    # start the audio-video streamer with the given config
    audio_video_streamer = AudioVideoStreamer(config, log_worker.log_queue)
    # audio_video_streamer.start()

    return audio_video_streamer, log_worker, thread
