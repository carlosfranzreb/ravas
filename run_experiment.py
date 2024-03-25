"""
Experiment with different configs. You can define lists of values for any config
parameter, and the script will run the experiment for all combinations of these
values. You can then analyse the logs somewhere else.
"""

from copy import deepcopy
import yaml
from stream_processing.main import main


def run_experiment(config: dict, params: dict, runtime: int) -> None:
    """
    Run the experiment with the given config and parameters.

    Args:
    - config: The base config.
    - params: A dictionary of parameters and their values. The keys should be the
        parameter names, and the values should be lists of values to try. For nested
        parameters, use a dot to separate the keys. Each parameter should have the same
        number of values. Experiments are run for each index.
    - runtime: The time to run each experiment. It should be enough for the audio model
        to start, which takes around 30 seconds.
    """
    n_values = len(next(iter(params.values())))
    for param_name, param_values in params.items():
        assert (
            len(param_values) == n_values
        ), "All parameters should have the same number of values"

    for value_idx in range(n_values):
        config_copy = deepcopy(config)
        for param_name, param_values in params.items():
            config_obj = config_copy
            param_name = param_name.split(".")
            for key in param_name[:-1]:
                config_obj = config_obj[key]
            config_obj[param_name[-1]] = param_values[value_idx]
        main(config_copy, runtime)


if __name__ == "__main__":
    config = yaml.safe_load(open("configs/run_experiment.yaml"))
    config["log_dir"] = "logs/experiment_logs"
    params = {
        "audio.processing_size": [3200, 9600],
        "audio.converter.vad.threshold": [0.0, 0.0],
    }
    run_experiment(config, params, 10)
