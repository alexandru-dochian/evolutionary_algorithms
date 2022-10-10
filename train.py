import os
from evoman_wrapper.EvomanWrapper import EvomanWrapper

"""
0. Removing video
"""
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


"""
1. Setup config
"""
EXPERIMENT_NAME = "SignalEAGroupA"
hidden_layers = 10
number_of_sensors = 20

config = {
    "config_type": "train",
    "hidden_layers": hidden_layers,
    "number_of_sensors": number_of_sensors,
    "experiment_name": EXPERIMENT_NAME,
    "environment_config": {
        "experiment_name": EXPERIMENT_NAME,
        "enemies": [1, 2, 3],
        "multiplemode": "yes",
        "playermode": "ai",
        "enemymode": "static",
        "level": 2,
        "speed": "fastest",
        "logs": "off",
        "player_controller": None,
    },
    "controller": {
        "instance": "NNController",
        "constructor": {
            "n_hidden": hidden_layers
        }
    },
    "evolutionary_algorithm": {
        "instance": "SignalEA",
        "constructor": {
            "number_of_genomes": (number_of_sensors + 1) * hidden_layers + (hidden_layers + 1) * 5,
            "number_of_parents": 25,
            "number_of_offsprings": 50,
            "max_generations": 500,
            "mutation_chance": 0.2,
            "mutation_signal_config": {
                "number_of_sine_functions": 10,
                "min_frequency": 0,
                "max_frequency": 100,
                "min_amplitude": -0.1,
                "amplitude_range": 0.2
            },
            "crossover_signal_config": {
                    "number_of_sine_functions": 2,
                    "min_frequency": 0,
                    "max_frequency": 50,
                    "min_amplitude": -1,
                    "amplitude_range": 2
            }
        }
    },
}

"""
2. Train
"""
evoman_wrapper = EvomanWrapper(config).train()