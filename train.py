from evoman_wrapper.EvomanWrapper import EvomanWrapper

"""
1. Setup config
"""
EXPERIMENT_NAME = "FirstEverExperiment"
hidden_layers = 10
number_of_sensors = 20

config = {
    "config_type": "train",
    "hidden_layers": 10,
    "number_of_sensors": 20,
    "experiment_name": EXPERIMENT_NAME,
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
    "controller": {
        "instance": "NNController",
        "constructor": {
            "n_hidden": 10
        }
    },
    "evolutionary_algorithm": {
        "instance": "EvolutionaryAlgorithm_1",
        "constructor": {
            "number_of_genomes": (number_of_sensors + 1) * hidden_layers + (hidden_layers + 1) * 5,
            "number_of_parents": 10,
            "number_of_offsprings": 10,
            "population_size": 20,
            "max_generations": 25,
            "mutation_chance": 0.5,
        }
    },
}

"""
2. Train
"""
evoman_wrapper = EvomanWrapper(config).train()
