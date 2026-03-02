import os
import pytest
import torch
from multiprocessing import Queue, Event
from stream_processing.utils import resolve_file_path

from stream_processing.models import KnnVC

"""
Test the KnnVC converter.

This test verifies:
- model loading
- Correct execution of the `convert_audio` function.
- Deterministic behavior when using deterministic inputs.
- That the output shape matches the expected dimensions.
"""


@pytest.fixture
def knnvc_config():
    return {
        "device": "cpu",
        "target_feats_path": "./target_feats/knnvc/john.pt",
        "n_cluster": 0,
        "n_neighbors": 4,
        "use_expressiveness": False,
        "prev_audio_queue": {"max_samples": 9600},
        "prev_ctx": {"use_previous_ctx": False, "max_samples": 0},
        "interpolator": {"n_samples": 450, "weight": 0.75},
    }


def _required_files_exist(config):
    model_size = config["prev_audio_queue"]["max_samples"]

    wavlm_path = resolve_file_path(f"onnx/wavlm_{model_size}.onnx")
    hifigan_path = resolve_file_path(f"onnx/hifigan_{model_size}.onnx")
    feats_path = resolve_file_path(config["target_feats_path"])

    return (
        os.path.exists(wavlm_path)
        and os.path.exists(hifigan_path)
        and os.path.exists(feats_path)
    )


@pytest.fixture(scope="function")
def knnvc(knnvc_config):
    if not _required_files_exist(knnvc_config):
        pytest.skip("Required ONNX models or target features not available")

    knnvc = KnnVC(
        name="test",
        config=knnvc_config,
        input_queue=Queue(),
        output_queue=Queue(),
        log_queue=Queue(),
        log_level="INFO",
        ready_signal=Event(),
    )

    return knnvc


def test_knnvc_init_loads_models(knnvc, knnvc_config):
    # Target features sanity
    assert knnvc.target_feats.ndim == 2
    assert knnvc.target_feats.shape[0] > 0

    assert knnvc.wavlm is not None  # WavLM should be loadable
    n_samples = knnvc_config["prev_audio_queue"]["max_samples"]
    torch.manual_seed(0)
    audio_random = torch.rand(1, n_samples, dtype=torch.float32)

    wavlm_out_1 = knnvc.wavlm.run(["output"], {"input": audio_random.numpy()})[0]
    wavlm_out_2 = knnvc.wavlm.run(["output"], {"input": audio_random.numpy()})[0]

    # WavLM should produce deterministic output for same input
    assert (
        wavlm_out_1.shape == wavlm_out_2.shape
    )  # wavlm output shape should be consistent
    assert (
        wavlm_out_1.dtype == wavlm_out_2.dtype
    )  # wavlm output dtype should be consistent
    assert torch.equal(
        torch.from_numpy(wavlm_out_1), torch.from_numpy(wavlm_out_2)
    )  # wavlm output should be identical for same input

    # HiFiGAN should be loadable and produce output of expected shape
    assert knnvc.hifigan is not None  # HiFiGAN should be loadable

    hifigan_input_size = (n_samples // 320) - 1
    hifigan_in = torch.randn((1, hifigan_input_size, 1024), dtype=torch.float32)

    hifigan_out_1 = knnvc.hifigan.run(["output"], {"input": hifigan_in.numpy()})[0]
    hifigan_out_2 = knnvc.hifigan.run(["output"], {"input": hifigan_in.numpy()})[0]

    # HiFiGAN should produce deterministic output for same input
    assert (
        hifigan_out_1.shape == hifigan_out_2.shape
    )  # hifigan output shape should be consistent
    assert (
        hifigan_out_1.dtype == hifigan_out_2.dtype
    )  # hifigan output dtype should be consistent
    assert torch.equal(
        torch.from_numpy(hifigan_out_1), torch.from_numpy(hifigan_out_2)
    )  # hifigan output should be identical for same input


def test_knnvc_convert_audio_output(knnvc, knnvc_config):
    n_samples = knnvc_config["prev_audio_queue"]["max_samples"]

    # Deterministic input (silence)
    audio_in = torch.zeros(n_samples, dtype=torch.int16)

    audio_out = knnvc.convert_audio(audio_in)

    assert isinstance(audio_out, torch.Tensor)
    assert audio_out.dtype == torch.int16
    assert (
        int(audio_out.shape[0]) == int(audio_in.shape[0]) - 320
    )  # -320 due to wavlm's receptive field

    # Value safety
    assert torch.max(audio_out) <= 32767
    assert torch.min(audio_out) >= -32768

    # Sanity: model produced non-trivial output
    assert torch.any(audio_out != 0)

    torch.manual_seed(0)
    audio_random = torch.rand(n_samples, dtype=torch.float32)

    out_1 = knnvc.convert_audio(audio_random)
    out_2 = knnvc.convert_audio(audio_random)

    # Deterministic output for same input
    assert out_1.shape == out_2.shape
    assert out_1.dtype == out_2.dtype

    assert torch.equal(
        out_1, out_2
    )  # convert audio should produce identical output for same input
