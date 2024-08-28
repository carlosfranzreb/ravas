from torch.multiprocessing import freeze_support

from stream_processing.main_gui import main


if __name__ == '__main__':
    # NOTE `freeze_support()` is required when building EXEC:
    freeze_support()
    main()
