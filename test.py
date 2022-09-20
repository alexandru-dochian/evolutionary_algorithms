from evoman_wrapper.EvomanWrapper import EvomanWrapper

"""
1. Setup config
"""
EXPERIMENT_NAME = "FirstEverExperiment"
hidden_layers = 10
number_of_sensors = 20

config = {
    "config_type": "test",
    "hidden_layers": 10,
    "number_of_sensors": 20,
    "experiment_name": EXPERIMENT_NAME,
    "chosen_generation": 20,
    "environment_config": {
        "experiment_name": EXPERIMENT_NAME,
        "enemies": [2],
        "playermode": "ai",
        "enemymode": "static",
        "level": 2,
        "speed": "fastest",
        "logs": "off",
        "player_controller": None,
    },
    "evolutionary_algorithm": None,
    "controller": {
        "instance": "NNController",
        "constructor": {
            "n_hidden": 10
        }
    },
    "games": [
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
        {"enemies": [2]},
    ]
}

"""
2. Test
"""
evoman_wrapper = EvomanWrapper(config).test()
