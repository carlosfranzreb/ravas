
import sys
from argparse import ArgumentParser, Namespace

from PyQt6.QtWidgets import QApplication

from .gui.main_window import MainWindow
from .utils import kill_all_child_processes


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/onnx_models.yaml")
    return parser.parse_args()


def main():
    app = QApplication(sys.argv)

    # configure default values (will also be used when creating QSettings)
    # IMPORTANT: do this before starting the windows/dialogs, so that they can access the correct QSettings objects
    app.setOrganizationName('Deutsches Forschungszentrum für Künstliche Intelligenz GmbH (DFKI)')
    app.setOrganizationDomain("dfki.de")
    app.setApplicationName("VERANDA Audio Video Streamer")  # TODO name/description: should this be the same as in setup.py?
    app.setApplicationVersion('0.1.0')  # TODO read from setup.py

    args = parse_args()
    window = MainWindow(config_path=args.config)

    print('staring application (pid %s)' % (app.applicationPid()))

    try:
        window.show()
        exit_code = app.exec()
        print('exited normally with code ', exit_code, flush=True)
    except KeyboardInterrupt:
        print('exited due to KeyboardInterrupt!', flush=True)
        exit_code = 0
        try:
            window.close()
        except:
            pass
    finally:
        print('stop streaming (if running)...', flush=True)
        window.stopStreaming()

    # without this, there may occur weird QThread messages in the shell on exit
    app.deleteLater()

    app.closeAllWindows()
    app.exit(exit_code)

    kill_all_child_processes(verbose=True)

    print('exit program now!', flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
