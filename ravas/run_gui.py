import traceback

from torch.multiprocessing import freeze_support

from stream_processing.main_gui import main
from stream_processing.utils import kill_all_child_processes


if __name__ == "__main__":
    has_error = False
    try:
        # NOTE `freeze_support()` is required when compiling executable for windows, for more details see
        #      https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
        freeze_support()

        main()
    except KeyboardInterrupt:
        pass
    except SystemExit as sys_exc:
        if sys_exc.code != 0:
            print("UNCLEAN SYSTEM EXIT (exit code: %s):" % sys_exc.code)
            traceback.print_exc()
            has_error = True
    except Exception:
        print("CRITICAL ERROR:")
        traceback.print_exc()
        has_error = True
    if has_error:
        print("\nTerminating remaining processes:")
        kill_all_child_processes(verbose=True)
