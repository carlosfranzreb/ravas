from argparse import ArgumentParser
import yaml

from stream_processing.main import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="ravas/configs/onnx_models_ui.yaml"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config, runtime=12)
