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
        parameters, use a dot to separate the keys. We allow only one level of nesting.
        Each parameter should have the same number of values, as we will run the
        experiment for each combination of values once.
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
            param_name = param_name.split(".")
            if len(param_name) == 1:
                config_copy[param_name[0]] = param_values[value_idx]
            else:
                config_copy[param_name[0]][param_name[1]] = param_values[value_idx]

        main(config_copy, runtime)


if __name__ == "__main__":
    config = yaml.safe_load(open("configs/default.yaml"))
    config["log_dir"] = "logs/proc_vs_buffer"
    params = {
        "audio.processing_size": [1600, 1600, 1600, 1600],
        "audio.record_buffersize": [400, 800, 1200, 1600],
    }
    run_experiment(config, params, 180)
