import json
import logging
import os
import time
from argparse import ArgumentParser, RawTextHelpFormatter
import textwrap

from torch.multiprocessing import Queue, Process
from typing import List, Dict, Optional

import moderngl_window
from moderngl_window import WindowConfig

from .opengl.main_render import GLTFRenderer
from ...dist_logging import worker_configurer


def test_render_recorded_blendshapes_to_queue(model_path: str, blendshapes_data_path: str, headless: bool = True):

    config = init_render_window_and_load_data(model_path, blendshapes_data_path, headless)
    win: GLTFRenderer = config.wnd

    # TEST will render the images into the queue:
    queue = Queue()
    win._DEBUG_enable_render_to_queue(queue)

    start = time.time()
    moderngl_window.run_window_config_instance(config)
    print('duration: ', time.time() - start)

    # FIXME need to empty queues "manually"
    #       ... maybe because queue is filled & emptied from same process/thread?
    #       ... instead of empty() using qsize() seems to work in this case,
    #           even though docs state that it is "imprecise"...
    while not queue.empty() or queue.qsize() > 0:
        queue.get()

    print('queue size: ', queue.qsize())


def run_renderer_window(model_path: str, blendshapes_data_path: Optional[str], headless: bool = True):

    config = init_render_window(headless)
    win: GLTFRenderer = config.wnd
    win.load_model(model_path)
    if blendshapes_data_path:
        win.load_recorded_blend_shapes(blendshapes_data_path)

    win.stop_on_data_end = False
    win.enable_stats(True)

    start = time.time()
    moderngl_window.run_window_config_instance(config)
    print('duration: ', time.time() - start)


def test_render_recorded_blendshapes_in_process(model_path: str, blendshapes_data_path: str, out_dir: str, headless: bool = True):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('storing data rendered images to ', the_out_dir, flush=True)

    in_queue = Queue()
    out_queue = Queue()
    app_args = {
        'model_path': model_path,
        'input_queue': in_queue,
        'output_queue': out_queue,
        'run_headless': headless,
    }
    render_app = Process(target=start_renderer, kwargs=app_args, name='render_app')
    render_app.start()

    rec_data = load_blendshapes_data(blendshapes_data_path)
    size = len(rec_data)
    start = time.time()
    failed = []

    # wait for "ready" signal:
    star_val = out_queue.get(timeout=10)
    assert star_val is False, "did not expected 'ready' signal (value False)"

    for i, d in enumerate(rec_data):
        # in_queue.put(json.dumps(d))
        in_queue.put(d)
        rendered = out_queue.get()
        if rendered:
            print(f'  rendered image ({i+1} / {size}), size (bytes): ', len(rendered), flush=True)
            # out_path = os.path.join(out_dir, f"queue_scene_{i+1}.jpg")
            # with open(out_path, 'bw') as f:
            #     f.write(rendered)
        else:
            print('  ERROR did not receive rendered data!', flush=True)
            failed.append((i, rendered))

    print('failed receiving (index, value): ', failed)
    print('duration: ', time.time() - start)
    print('sending "stop" signal...', flush=True)
    in_queue.put(None)
    time.sleep(1)

    print('queue size (1): ', out_queue.qsize(), flush=True)

    # FIXME need to empty queues "manually"
    #       ... instead of empty() using qsize() seems to work in this case,
    #           even though docs state that it is "imprecise"...
    while not out_queue.empty() or out_queue.qsize() > 0:
        print('dequeueing -> ', out_queue.get())

    print('queue size (2): ', out_queue.qsize(), flush=True)

    render_app.terminate()


def test_store_images_for_recorded_blendshapes(model_path: str, blendshapes_data_path: str, out_dir: str, headless: bool = True):

    print('storing data rendered images to ', the_out_dir, flush=True)

    config = init_render_window_and_load_data(model_path, blendshapes_data_path, headless)
    win: GLTFRenderer = config.wnd

    win.enable_render_and_store_images(out_dir)

    start = time.time()
    moderngl_window.run_window_config_instance(config)
    print('duration: ', time.time() - start)


def init_render_window_and_load_data(model_path: str, blendshapes_data_path: str, headless: bool = False) -> WindowConfig:
    config = init_render_window(headless)
    win: GLTFRenderer = config.wnd
    win.load_model(model_path)
    win.load_recorded_blend_shapes(blendshapes_data_path)
    return config


def init_render_window(headless: bool, log_level=None):

    if log_level is not None:
        GLTFRenderer.log_level = log_level

    if headless:
        GLTFRenderer.hidden_window_framerate_limit = -1
        GLTFRenderer.visible = False
        GLTFRenderer.is_headless = True
        win_args = ("--window", "headless")
    else:
        GLTFRenderer.hidden_window_framerate_limit = 30
        GLTFRenderer.visible = True
        # GLTFRenderer.cursor = True  # see below w.r.t. win_args
        GLTFRenderer.is_headless = False
        # IMPORTANT: must supply an args here, because if set to None, it will read the args from the command line
        #            which would throw an error / unsupported args, if we run our main program with any args
        win_args = ("--cursor", "True")  # <- set the cursor via args, so that it is not empty


    config = moderngl_window.create_window_config_instance(GLTFRenderer, args=win_args)
    return config


def start_renderer(
        model_path: str,
        input_queue: Queue,
        output_queue: Queue,
        run_headless: bool,
        log_queue: Optional[Queue] = None,
        log_level: Optional[str] = None,
):
    if log_queue:
        worker_configurer(log_queue, log_level if log_level else 'INFO')

    try:
        config = init_render_window(run_headless, log_level)
        win: GLTFRenderer = config.wnd
        win.load_model(model_path)

        win.enable_render_queue_io(input_queue, output_queue)
        output_queue.put(False)  # <- send "ready signal"
        moderngl_window.run_window_config_instance(config)
    except Exception as exc:
        print('ERROR for open gl renderer', exc, flush=True)
        logging.getLogger().error('ERROR for open gl render: %s', exc, exc_info=exc, stack_info=True)


def load_blendshapes_data(data_path: str) -> List[Dict[str, any]]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    """ run with --help to show available options """

    # create args-parser that shows help only with arg "--help" (i.e. disable "-h" in order to use for own args):
    parser = ArgumentParser(description="run OpenGL based avatar renderer", add_help=False, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--help", action="help", help=textwrap.dedent("show this help message and exit"))

    default_avatar_path = os.path.realpath(os.path.join(__file__, '../../../../../rpm/public/avatar_1_f.glb'))
    default_blendshapes_dir = os.path.realpath(os.path.join(__file__, '../../../../logs/<DATE TIME>/detectionLog.json'))
    default_out_dir = os.path.realpath('output')
    default_mode = 'win'
    parser.add_argument(
        "-h", "--headless",
        action="store_true",
        help=textwrap.dedent("run in headless mode (without visible window)")
    )
    parser.add_argument(
        "-a", "--avatar",
        type=str,
        help=textwrap.dedent("path to the avatar model file (*.glb)\n"
                             "DEFAULT:\n  "+default_avatar_path),
        default=default_avatar_path
    )
    parser.add_argument(
        "-b", "--blendshapes",
        type=str,
        help=textwrap.dedent("path to the recorded blendshapes:\n"
                             "can be recorded by running the stream-processor with configuration\n"
                             "`video.converter.detection_log: detectionLog.json.`\n"
                             "The recorded JSON file will be in the log directory:\n"
                             "EXAMPLES:\n  "+default_blendshapes_dir+"\n  logs/<DATE TIME>/detectionLog.json"),
        required=False
    )
    parser.add_argument(
        "-m", "--mode",
        choices=['win', 'test_store', 'test_queue', 'test_process'],
        help=textwrap.dedent("mode for running the renderer:\n"
                             " * `win`: open renderer window with stats feedback (also allows to control camera view)\n"
                             " * `test_store`: test storing the rendered images as files (requires: `-b BRLENDSHAPES` and `-o OUT`)\n"
                             " * `test_queue`: test rendering images into an multiprocessing.Queue (requires `-b BRLENDSHAPES`)\n"
                             " * `test_process`: test running the renderer in another process and storing the rendered images as files (requires `-b BRLENDSHAPES` and `-o OUT`)\n"
                             "DEFAULT:\n  "+default_mode),
        default=default_mode
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        help=textwrap.dedent("path to for the output directory for storing rendered images (required for mode `-m test_store` and `-m test_process`)\n"
                             "DEFAULT:\n  " + default_out_dir),
        default=default_out_dir
    )
    args = parser.parse_args()

    is_headless = args.headless

    # set to a log-dir that has recorded blendshapes (-> record via configuration `video.converter.detection_log`):
    the_blendshapes_data_path = args.blendshapes

    the_model_path = args.avatar
    print('using avatar from:\n', the_model_path, flush=True)

    # path for storing rendered images:
    the_out_dir = args.out

    def validate_b():
        if not the_blendshapes_data_path or not os.path.exists(the_blendshapes_data_path) or not os.path.isfile(the_blendshapes_data_path):
            if not the_blendshapes_data_path:
                reason = 'not specified'
            elif not os.path.exists(the_blendshapes_data_path):
                reason = 'file does not exist at ' + os.path.realpath(the_blendshapes_data_path)
            else:
                reason = 'path is not file, at ' + os.path.realpath(the_blendshapes_data_path)
            raise ValueError('invalid path for recorded blendshapes (option `-b`): '+reason)

    the_mode = args.mode
    match the_mode:
        case 'win':
            run_renderer_window(the_model_path, the_blendshapes_data_path, headless=is_headless)
        case 'test_store':
            validate_b()
            test_store_images_for_recorded_blendshapes(the_model_path, the_blendshapes_data_path, out_dir=the_out_dir, headless=is_headless)
        case 'test_queue':
            validate_b()
            test_render_recorded_blendshapes_to_queue(the_model_path, the_blendshapes_data_path, headless=is_headless)
        case 'test_process':
            validate_b()
            test_render_recorded_blendshapes_in_process(the_model_path, the_blendshapes_data_path, out_dir=the_out_dir, headless=is_headless)
        case _:
            raise ValueError(f'unknown mode "{the_mode}": use --help to list available modes')
