import unittest
import torch

from ..stream_processing.models.knnvc.converter import rms


class TestRMS(unittest.TestCase):
    def test_rms_perfect_fit(self):
        """
        Test case where the audio length is covered by the hop and frame lengths.
        Frame 1: [1.0, 2.0, 3.0] -> RMS: 2.1602
        Frame 2: [3.0, 4.0, 5.0] -> RMS: 4.0825
        """
        audio = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        frame_length = 3
        hop_length = 2
        expected_rms = torch.tensor([2.1602, 4.0825])
        rms_output = rms(audio, frame_length, hop_length)
        self.assertTrue(torch.allclose(rms_output, expected_rms, atol=1e-4))

    def test_rms_without_padding(self):
        """
        Test case where the whole audio length is convered by the first frame. There
        should be a second frame according to the hop length, but we skip it.
        Frame 1: [1.0, 2.0, 3.0] -> RMS: 2.1602
        """
        audio = torch.tensor([1.0, 2.0, 3.0])
        frame_length = 3
        hop_length = 2
        expected_rms = torch.tensor([2.1602])
        rms_output = rms(audio, frame_length, hop_length)
        self.assertTrue(torch.allclose(rms_output, expected_rms, atol=1e-4))

    def test_rms_single_frame(self):
        """
        Test case with a single frame, where the hop and frame lengths are larger than
        the audio length. Here, the RMS value is the audio's RMS.
        Frame 1: [1.0] -> RMS: 1.0
        """
        audio = torch.tensor([1.0])
        frame_length = 4425
        hop_length = 134
        expected_rms = torch.tensor([1.0])
        rms_output = rms(audio, frame_length, hop_length)
        self.assertTrue(torch.allclose(rms_output, expected_rms, atol=1e-4))

    def test_rms_large_audio(self):
        """Test case with a large audio."""
        audio = torch.arange(1000, dtype=torch.float32)
        frame_length = 100
        hop_length = 50
        expected_rms = torch.sqrt(
            torch.mean(audio.unfold(0, frame_length, hop_length).pow(2), dim=-1)
        )
        rms_output = rms(audio, frame_length, hop_length)
        self.assertTrue(torch.allclose(rms_output, expected_rms))


if __name__ == "__main__":
    unittest.main()
