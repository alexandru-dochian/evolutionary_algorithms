import os

from evoman_wrapper.EvomanWrapper import EvomanWrapper

"""
0. Removing video
"""
headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


"""
1. Setup config
"""
EXPERIMENT_NAME = "HighFrequency"
hidden_layers = 10
number_of_sensors = 20

config = {
    "config_type": "test",
    "hidden_layers": hidden_layers,
    "number_of_sensors": number_of_sensors,
    "experiment_name": EXPERIMENT_NAME,
    "chosen_generation": 100,
    "environment_config": {
        "experiment_name": EXPERIMENT_NAME,
        "enemies": [1, 2, 3],
        "multiplemode": "yes",
        "playermode": "ai",
        "enemymode": "static",
        "level": 2,
        "speed": "normal",
        "logs": "off",
        "player_controller": None,
    },
    "evolutionary_algorithm": None,
    "controller": {
        "instance": "NNController",
        "constructor": {
            "n_hidden": hidden_layers
        }
    },
    "games": [
        {"enemies": [1, 2, 3]},
    ]
}

"""
2. Test
"""
evoman_wrapper = EvomanWrapper(config).test()
