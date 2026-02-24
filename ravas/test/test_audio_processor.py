import unittest
import os
import torch
import tempfile

from multiprocessing import Event, Queue
from unittest.mock import patch

from ravas.stream_processing.audio_processor import get_device_idx, get_wav_obj, AudioProcessor, ProcessingSyncState


class TestGetDeviceIdx(unittest.TestCase):
    '''
    Test for get_device_idx function
    The get_device_idx should return the index for a given idx/name for either input/output device.
    If its not avaible or doesn't exist it should raise ValueError.
    '''
    @patch("ravas.stream_processing.audio_processor.sd.query_devices")
    def test_get_device_idx_by_index(self, mock_query_devices):
        mock_query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
        self.assertEqual(get_device_idx(0, is_input=True), 0) 
        self.assertEqual(get_device_idx(1, is_input=False), 1) 

        with self.assertRaises(ValueError):
            get_device_idx(3,is_input=True)
            get_device_idx(4,is_input=False)

    @patch("ravas.stream_processing.audio_processor.sd.query_devices")
    def test_get_device_idx_by_name(self, mock_query_devices):
        mock_query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
        ]
        self.assertEqual(get_device_idx("Mic", is_input=True), 0) 
        self.assertEqual(get_device_idx("Speaker", is_input=False), 1) 

        with self.assertRaises(ValueError):
            get_device_idx("Unknown",is_input=True)
            get_device_idx("Exists",is_input=False)

class TestGetWavObj(unittest.TestCase):
    '''
    Test for get_wav_obj function
    Should return an wav object for the given path with the correct channel and sampling_size
    '''
    def test_get_wav_obj_creates_wav_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "audio.wav")

            wav = get_wav_obj(path, sample_rate=16000)

            self.assertTrue(os.path.exists(path))
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 16000)

            wav.close()

class TestAudioProcessorInit(unittest.TestCase):
    '''
    Test for the init of AudioProcessor class
    '''
    @patch("ravas.stream_processing.audio_processor.get_device_idx")
    def test_init_video_file(self, mock_get_device_idx):
        '''
        Test for loading a video file and storing it
        - Input and Output device should be none as they are not being used
        - store file path should exist
        '''
        config = {
            "video_file": "video.mp4",
            "input_device": 0,
            "output_device": 1,
            "store": True,
            "log_dir": "/tmp",
            "sampling_rate": 16000,
            "max_unsynced_time": 0.1,
        }

        processor = AudioProcessor(
            config=config,
            audio_sync_state=ProcessingSyncState(),
            external_sync_state=ProcessingSyncState(),
            pipeline_sync_state=Event(),
            log_queue=Queue(),
            log_level="INFO"
        )

        mock_get_device_idx.assert_not_called()

        self.assertIsNone(processor.input_device)
        self.assertIsNone(processor.output_device)
        self.assertTrue(processor.store_path.endswith("audio.wav"))

    @patch("ravas.stream_processing.audio_processor.get_device_idx")
    def test_init_without_video_file(self, mock_get_device_idx):
        '''
        Test without a given video file
        - Input and Output device should initialized
        - store path shouldn't exist
        '''
        mock_get_device_idx.side_effect = [0,1]
        config = {
            "video_file": None,
            "input_device": "Mic",
            "output_device": "Speaker",
            "store": False,
            "log_dir": "/tmp",
            "sampling_rate": 16000,
            "max_unsynced_time": 0.1,
        }

        processor = AudioProcessor(
            config=config,
            audio_sync_state=ProcessingSyncState(),
            external_sync_state=ProcessingSyncState(),
            pipeline_sync_state=Event(),
            log_queue=Queue(),
            log_level="INFO"
        )

        self.assertEqual(processor.input_device,0)
        self.assertEqual(processor.output_device,1)
        self.assertFalse(hasattr(processor, "store_path"))

class TestAudioProcessorRead(unittest.TestCase):
    @patch("ravas.stream_processing.audio_processor.pyaudio.PyAudio")
    @patch("ravas.stream_processing.audio_processor.batchify_input_stream")
    def test_read_stream(self, mock_batchify, mock_pyaudio):
        num_chunks = 1
        record_buffersize = 480
        chunks = [
                (
                    (torch.tensor([i]), torch.full((record_buffersize,) ,i ,dtype=torch.int16)),
                    (torch.empty(0), torch.empty(0)),
                )
                for i in range(num_chunks)
        ]
        chunks.append(((None, None), (torch.empty(0), torch.empty(0))))
        chunks.append(((None, None), (torch.empty(0), torch.empty(0))))
        mock_batchify.side_effect = chunks

        config = {
            "video_file": None,
            "input_device": 0,
            "output_device": 1,
            "record_buffersize": record_buffersize,
            "processing_size": record_buffersize,
            "store": True,
            "log_dir": "/tmp",
            "sampling_rate": 16000,
            "max_unsynced_time": 0.1,
        }

        processor = AudioProcessor(
            config=config,
            audio_sync_state=ProcessingSyncState(),
            external_sync_state=ProcessingSyncState(),
            pipeline_sync_state=Event(),
            log_queue=Queue(),
            log_level="INFO"
        )

        processor.queues.ready.set()

        processor.read()

        result = []

        while not processor.queues.input_queue.empty():
            t, data = processor.queues.input_queue.get_nowait()
            if data is not None:
                result.append((t, data))

        self.assertEqual(len(result), num_chunks)
        for i, (t, data) in enumerate(result):
            self.assertEqual(t.item(), i)
            self.assertTrue(torch.all(data == i))

class TestAudioProcessorWrite(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()