# RAVAS 🦑 - Real-time Audiovisual Anonymization System

System to anonymize audio and video in real-time and locally.
With [ReadyPlayerMe's avatar](https://readyplayer.me/) and Mimi-VC, our own real-time voice converter based on [Kyutai's speech tokenizer](https://github.com/kyutai-labs/moshi), it runs with a latency of 0.13 seconds in my Macbook pro M1.
RAVAS can also be used to anonymize videos, emulating the real-time scenario, to perform experiments.
We built this at [DFKI](https://www.dfki.de/web) for the [AnonymPrevent](https://www.tu.berlin/en/qu/research/current-past-projects/laufende-projekte/anonymprevent) and [VERANDA](https://www.tu.berlin/en/qu/research/current-past-projects/laufende-projekte/veranda) research projects.

You can read more technical details in [this blog post](https://carlosfranzreb.github.io/ravas).

<https://github.com/user-attachments/assets/9f7c5801-3a06-4852-a818-059c51e61592>

## Contributors

Here are the most important contributions by the three colleagues that have helped me develop RAVAS, in chronological order.
They have all done much more than what is stated here, these are just the highlights.
Huge thanks to them for their effort.

1. [@phipi-a](https://github.com/phipi-a): implemented the first version of the avatar the multi-threaded and sync-preserving architecture.
2. [@russaa](https://github.com/russaa): implemented the Windows packaging, the GUI and a faster and more robust version of the avatar.
3. [@HuangJ98](https://github.com/HuangJ98): implemented chunking strategies for kNN-VC and [private kNN-VC](https://github.com/carlosfranzreb/private_knnvc).

## Citation

```latex
@inproceedings{franzreb24_spsc,
  title     = {Towards Audiovisual Anonymization for Remote Psychotherapy: a Subjective Evaluation},
  author    = {Carlos Franzreb and Arnab Das and Hannes Gieseler and Eva Charlotte Jahn and Tim Polzehl and Sebastian Möller},
  year      = {2024},
  booktitle = {4th Symposium on Security and Privacy in Speech Communication},
  pages     = {102--110},
  doi       = {10.21437/SPSC.2024-17},
}
```

---------

__TOC__
<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:0 orderedList:0 -->

- [How to use the implementation](#how-to-use-the-implementation)
- [Project Structure](#project-structure)
    - [Notes](#notes)
- [Installation guide](#installation-guide)
- [Build Executable](#build-executable)
- [Usage](#usage)
    - [Console Program](#console-program)
    - [GUI](#gui)
- [Avatar Anonymizer](#avatar-anonymizer)
    - [Python-based Avatar Renderer (Default)](#python-based-avatar-renderer-default)
    - [Web-based Avatar Renderer (Legacy)](#web-based-avatar-renderer-legacy)
    - [Changing The Avatar](#changing-the-avatar)
- [Development](#development)
    - [Add new Setting in GUI](#add-new-setting-in-gui)

<!-- /TOC -->

---------

## How to use the implementation

An example is provided in the ./example folder. The general workflow is as follows:
1. Define the `init_callback` function for audio or video, where you can initialize all objects required for the `callback` function. Return these objects to use them in the callback.
2. Define the final `callback` function, which receives the data, corresponding timestamps, and the list of objects returned from the `init_callback` function.
3. Create an `AudioVideoStreamer` object and set all parameters like the sampling rate, audio/video device, etc., along with the `init_callback` and the `callback` function.
4. Start the streamer with the `start()` function.
5. Wait for a KeyboardInterrupt or some other input.
6. Stop the streamer with the `stop()` function.

## Project Structure

The project is structured as follows:

- **Processor.py**
    - `Processor`: Abstract class providing basic functionality to process data, such as synchronization (`sync`)
                   and callback function handling (`process`). The `read_input_stream` and `write_output_stream` functions
                   are abstract and must be implemented in the child class. Since each function will be started in a new process,
                   the class is only allowed to have attributes that are pickable.
    - `ProcessingQueues`: Provides all necessary multiprocessing queues for the `Processor` class.
    - `ProcessingSyncState`: Provides all necessary multiprocessing values to synchronize different processes.
    - `ProcessorProcessHandler`: Has only two functions used to start and stop the functions of the `Processor` class in a new process.
- **AudioProcessor.py**
    - `AudioProcessor`: Extends the `Processor` class and provides all necessary functions to process audio data.
       Because each function will be started in a new process, the class is only allowed to have attributes that are pickable.
- **VideoProcessor.py**
    - `VideoProcessor`: Extends the `Processor` class and provides all necessary functions to process video data.
      Because each function will be started in a new process, the class is only allowed to have attributes that are pickable.
- **AudioVideoStreamer.py**
    - `AudioVideoStreamer`: Provides an API to combine the `AudioProcessor` and `VideoProcessor` classes and share their sync state.

### Notes

In the `AudioVideoStreamer` class, an `AudioProcessor`, `VideoProcessor`, `ProcessingSyncState`, and `ProcessingQueues` object
is created for each processor. The queues are used in the corresponding processor to exchange data between the processes.
The `ProcessingSyncState` object is used to synchronize the different processors (`AudioProcessor` and `VideoProcessor`).
A `ProcessorProcessHandler` object is used to start and stop the processes of the `AudioProcessor` and `VideoProcessor` class.
When calling the `start` function of the handler, each function of the `Processor` gets called in its own process
(`read`, `process`, `sync`, `write`).

The different processes communicate through the multiprocessing queues and synchronize through the multiprocessing values
provided in the sync state. Due to multiprocessing, the `Processor` class is only allowed to have attributes that are pickable.
Therefore, the callback and `init_callback` functions must be defined outside the main function (see example).

To reduce latency between the streams, the data is converted into torch tensors. Torch tensors are stored in shared memory
and can be accessed by all processes through the queues without copying the data. For numpy arrays, the data has to be
pickled and unpickled to be sent through the queues, which is very slow.

## Installation guide

__MacOS__

Copy-paste these lines in a terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install portaudio
brew install blackhole-2ch

git clone https://github.com/carlosfranzreb/ravas.git
cd ravas
bash ./install_macos.sh
```

See also the [prerequisites](USAGE.md#prerequisites) section, in the `USAGE.md` file, for other OS prerequisites w.r.t.
to audio and camera/video processing.

## Build Executable

For building / packaging the `python` application as a standalone executable see [ravas/BUILD.md](ravas/BUILD.md).

## Usage

### Console Program

for starting the main app as a console program, change to directory `ravas/` and run

```bash
python -m run_main
```

You can specify a configration file via `--config <file>`, see [ravas/configs](ravas/configs) for configuration examples.

### GUI

for starting the main app with a GUI, change to directory `ravas/` and run

```bash
python -m run_gui
```

See [USAGE.md](USAGE.md) for more usage information for the GUI.

---

## Avatar Anonymizer

The avatar anonymizer is composed of the [mediapipe's face-landmarker][4] solution for detecting face expressions and head movement
from the camera / video images, and then using the result to render an avatar model (in `*.glb` format: binary glTF 2.0 format).

The avatar is now rendered with the new OpenGL renderer, using the [moderngl][5] and [moderngl-window][6] wrappers for OpenGL in `python`, see `ravas/stream_processing/models/avatar/opengl_runner.py` and `ravas/stream_processing/models/avatar/opengl/*`.
Note that to avoid unnecessary duplication, the renderer currently uses the web app's avatar model files in `rpm/dist/chrome-extension/*.glb` or `rpm/public/*.glb`

Previously, we used a React rendered based on [this repository](https://github.com/srcnalt/rpm-face-tracking).
The new one is faster and we have since removed the old renderer; check the commits if you want to try it out.

The basic configuration for the avatar renderer via the YAML configuration files (as video converter) are

```yml
video:
    # ...
    converter:
        cls: stream_processing.models.Avatar
        # select renderer for avatar: opengl (DEFAULT) | browser
        avatar_renderer: opengl
        # if TRUE, will show the rendering app window (e.g. for DEBUG purposes); for browser renderer, will show the browser window
        show_renderer_window: false
        # the avatar model to be used for rendering (see `rpm/public/*.glb`
        #  * the value `default_avatar.glb` will be mapped to the first available / default avatar
        #  * for the browser renderer, the values should be file names or relative to the immediate parent directory
        #    (i.e. no absolute file paths); the opengl renderer can also handle absolute file paths
        avatar_uri: ./default_avatar.glb
        # OPTIONAL store mediapipe's detection results for facial expression / head movement to an array in a JSON file in the log-dir:
        # NOTE that the last entry in the array will be an empty object, i.e. without "blendshapes" field!
        detection_log: detectionLog.json
        # for browser renderer: if TRUE, start Chrome Browser automatically via Selenium webdriver
        start_chrome_renderer: true
        # for browser renderer: if TRUE, use Chrome Web Extension, if FALSE start a web server for serving web app as a website
        use_chrome_extension: true
```

### Python-based Avatar Renderer (Default)

The `python` based avatar renderer is implemented using the Python [moderngl][5] wrapper for OpenGL and running the
renderer in a [moderngl-window][6].

> TODO: currently the implementation uses the default `pyglet` integration of moderngl-window.
> There is also a `PyQT5` integration in moderngl-window: when `PyQT6` integration becomes available, we should
> switch to that, so that we use the same window-system as the main app's GUI.

For testing and debugging, the `python`-based avatar renderer can be started in _standalone_ mode with

```bash
python -m stream_processing.models.avatar.opengl_runner
```

There are several commend line options available (use `--help`), e.g. for selecting a specific avatar model (`-a` / `--avatar`).

It is also possible to _"play back"_ previously recorded face expression & head movements ("blendshapes"):

 1. configure the main app, to record `mediapipe`'s face-landmarker results in a JSON file

    ```yml
    video:
        # ...
        converter:
            cls: stream_processing.models.Avatar
            # ...
            # store mediapipe's detection results for facial expression / head movement to an array in a JSON file in the log-dir:
            # NOTE that the last entry in the array will be an empty object, i.e. without "blendshapes" field!
            detection_log: detectionLog.json
    ```

 2. run the main app and store the detection results
 3. find the log-dir where the detection results were stored and run the `python` avatar renderer with the argument `-b` / `--blendshapes`:

    ```bash
    python -m stream_processing.models.avatar.opengl_runner -b <path to recorded detection results in log-directory>
    ```

### Changing The Avatar

The avatars are stored in a GLB files in the `rpm/public/` folder, and are used by the default renderer as well as the
legacy web-based renderer (i.e. in  the React app (`rpm/src/App.tsx`)).

The naming scheme for the avatar files is `avatar_<number>_<gender: f | m>.glb`

If you want to change or add avatars:

1. create an avatar on the [Ready Player Me][1] website  
   _(currently this is free; you do not need an account either)_
2. get the download ID / link for created avatar, e.g. something like <https://models.readyplayer.me/6460d95f9ae10f45bffb2864.glb>
3. __IMPORTANT__ the avatar model files __must__ include _morph target_ definitions for `ARKit`
   (i.e. `morphTargets=ARKit`, see [Ready Player Me REST API docs][2]),
   and the texture quality __should__ be set to high (i.e. `quality=high` or `textureAtlas=1024`):  
   add these query-parameters to the download link for the avatar, e.g. <https://models.readyplayer.me/6460d95f9ae10f45bffb2864.glb?morphTargets=ARKit&textureAtlas=1024>
4. rename the `*.glb` avatar file (see naming scheme above) and place it in the project folder `rpm/public/`
5. you should also rebuild the web app for the avatar rendering (see [rpm/README.md][3])

## Audio Anonymizer

We currently support 2 Anonymizers: KnnVC and MimiVC.

### Adding previous context

Since the input and output are constrained by ONNX, the additional context must be included in the total processing size.
The currently available total processing sizes are __3200, 4800, and 9600__.
There are two ways to enable this option and adjust the previous context size:

1. inside the GUI under the __Advanced Settings__
2. in the config files manually

    ```yml
    audio:
        # ...
        converter:
            # ...
            prev_ctx:
                use_previous_ctx: false
                max_samples: 0
    ```

In the pre-release v0.6, the checkpoints include an additional 320-sample input, which is only required for the default computation to compensate for feature loss during WavLM. If you are already using a larger previous context, you can ignore the WavLM loss.

---

## Development

### Add new Setting in GUI

A short guide on how to add a new setting (i.e. a configuration item from the `*.yaml` based config) in the GUI
in [ravas/stream_processing/gui/](ravas/stream_processing/gui).

The current implementation uses `ConfigurationItem`s to represent configuration properties in the GUI. They define

 1. the "path" to the configuration property within the YAML configuration (e.g. something like `["video", "use_video"]`)
 2. specify the allowed configuration values:  \
    these (plus, if necessary, additional helper functions) allow to validate the current configuration, and thus allow
    feedback to users in case there is a misconfiguration.

To add a new setting in the GUI:

1. __New Item:__ create a new `ConfigurationItem` in [gui/config_items.py](ravas/stream_processing/gui/config_items.py):  
   add this to the module constant `CONFIG_ITEMS`.

2. __Validation:__ if the new setting's _valid value validation_ depends on other settings:

   - e.g. a setting that is only relevant in case the avatar video-converter is selected, or if audio processing is enabled,
     then add a `is_ignore_validation(current_config: Dict) -> bool` helper function to the configuration item.  
   - As example, see `_do_set_ignore_validation_helpers()` where this is done for several item defined in `CONFIG_ITEMS`
     (you may modify this function to add your own adjustments).

3. __Add to GUI:__ then add a GUI widget for the new item to `ConfigDialog` in [gui/config_dlg.py](ravas/stream_processing/gui/config_dlg.py)
   - currently, this is done in the class' construction `__init__()`
   - the class provides several helper functions to create some default widgets
     - `_createCheckBoxFor(..)`: for creating a check-box widget (usually used for boolean settings)
     - `_createComboBoxFor(..)`: for creating a combo-box widget (usually used for settings with a list of valid values)
     - ... as well as some more intricate widgets like `_createSliderFor(..)` for creating slider control for number settings
   - NOTE that the GUI somewhat duplicates the validation logic in order to enable/disable configuration item widgets:  
     see for example local functions `_updateAvatarEnabled()`, `_updateAvatarRendererSelected()`,
     `_set_audio_widgets_enabled()`, and `_set_video_widgets_enabled()`

[1]: https://readyplayer.me/avatar
[2]: https://docs.readyplayer.me/ready-player-me/api-reference/rest-api/avatars/get-3d-avatars
[3]: rpm/README.md
[4]: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
[5]: https://moderngl.readthedocs.io/
[6]: https://pypi.org/project/moderngl-window/
[7]: https://react.dev/
[8]: https://pypi.org/project/selenium/
