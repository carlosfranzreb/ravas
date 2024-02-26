# STREAM_PROCESSING
# ===============

## How to use it
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
    - `Processor`: Abstract class providing basic functionality to process data, such as synchronization (`sync`) and callback function handling (`process`). The `read_input_stream` and `write_output_stream` functions are abstract and must be implemented in the child class. Since each function will be started in a new process, the class is only allowed to have attributes that are pickable.
    - `ProcessingQueues`: Provides all necessary multiprocessing queues for the `Processor` class.
    - `ProcessingSyncState`: Provides all necessary multiprocessing values to synchronize different processes.
    - `ProcessorProcessHandler`: Has only two functions used to start and stop the functions of the `Processor` class in a new process.
- **AudioProcessor.py**
    - `AudioProcessor`: Extends the `Processor` class and provides all necessary functions to process audio data. Because each function will be started in a new process, the class is only allowed to have attributes that are pickable.
- **VideoProcessor.py**
    - `VideoProcessor`: Extends the `Processor` class and provides all necessary functions to process video data. Because each function will be started in a new process, the class is only allowed to have attributes that are pickable.
- **AudioVideoStreamer.py**
    - `AudioVideoStreamer`: Provides an API to combine the `AudioProcessor` and `VideoProcessor` classes and share their sync state.

### Notes

In the `AudioVideoStreamer` class, an `AudioProcessor`, `VideoProcessor`, `ProcessingSyncState`, and `ProcessingQueues` object is created for each processor. The queues are used in the corresponding processor to exchange data between the processes. The `ProcessingSyncState` object is used to synchronize the different processors (`AudioProcessor` and `VideoProcessor`). A `ProcessorProcessHandler` object is used to start and stop the processes of the `AudioProcessor` and `VideoProcessor` class. When calling the `start` function of the handler, each function of the `Processor` gets called in its own process (`read`, `process`, `sync`, `write`).

The different processes communicate through the multiprocessing queues and synchronize through the multiprocessing values provided in the sync state. Due to multiprocessing, the `Processor` class is only allowed to have attributes that are pickable. Therefore, the callback and `init_callback` functions must be defined outside the main function (see example).

To reduce latency between the streams, the data is converted into torch tensors. Torch tensors are stored in shared memory and can be accessed by all processes through the queues without copying the data. For numpy arrays, the data has to be pickled and unpickled to be sent through the queues, which is very slow.

## Installation guide

1. Install portaudio: `brew install portaudio`
2. Install the package: `pip install .`
3. Install a virtual audio loopback driver, e.g. `brew install blackhole-2ch`
