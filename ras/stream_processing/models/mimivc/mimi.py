from pathlib import Path

import torch
from torch import Tensor
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from moshi.models.compression import MimiModel as MimiModelOg
from moshi.quantization import SplitResidualVectorQuantizer

from .mimi_functional.transformer import StreamingTransformer
from .mimi_functional.seanet import SEANetEncoder as SEANetEncoderF
from .mimi_functional.seanet import SEANetDecoder as SEANetDecoderF
from .mimi_functional.mimi import MimiModel

DEVICE = "cpu"
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)

MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"

_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(
    filename: str = None,
    device: str = "cpu",
    num_codebooks: int = 8,
    og: bool = False,
) -> MimiModel:
    """Return a pretrained Mimi model, or unintialized if `filename` is None."""
    encoder = SEANetEncoderF(**_seanet_kwargs)
    decoder = SEANetDecoderF(**_seanet_kwargs)
    encoder_transformer = StreamingTransformer(device=device, **_transformer_kwargs)
    decoder_transformer = StreamingTransformer(device=device, **_transformer_kwargs)
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model_cls = MimiModelOg if og else MimiModel
    model = model_cls(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if filename is not None:
        if _is_safetensors(filename):
            state = load_file(filename, device=str(device))
            consume_prefix(state, "transformer")
            model.load_state_dict(state)
        else:
            pkg = torch.load(filename, "cpu")
            model.load_state_dict(pkg["model"])
    model.set_num_codebooks(num_codebooks)
    return model


def consume_prefix(state_dict: dict[str, Tensor], prefix: str) -> None:
    """
    Strip the prefix in state_dict in place, if any.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = list(state_dict.keys())
    for key in keys:
        key_arr = key.split(".")
        if prefix in key_arr:
            key_arr.remove(prefix)
            new_key = ".".join(key_arr)
            state_dict[new_key] = state_dict.pop(key)


def hf_get(
    filename: str | Path,
    hf_repo: str | None = None,
    check_local_file_exists: bool = False,
) -> Path:
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        parts = filename.removeprefix("hf://").split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename))
    elif filename.startswith("file://"):
        # Provide a way to force the read of a local file.
        filename = filename.removeprefix("file://")
        return Path(filename)
    elif hf_repo is not None:
        if check_local_file_exists:
            if Path(filename).exists():
                return Path(filename)
        return Path(hf_hub_download(hf_repo, filename))
    else:
        return Path(filename)


def init_mimi():
    mimi_weights = hf_get(MIMI_NAME, DEFAULT_REPO)
    mimi = get_mimi(mimi_weights, num_codebooks=8, device=DEVICE)
    batch_size = 1
    enc_state, tr_enc_state, tr_dec_state, dec_state = mimi._init_streaming_state(
        batch_size
    )
    return mimi, enc_state, tr_enc_state, tr_dec_state, dec_state
