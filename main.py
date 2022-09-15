import sys, os

sys.path.insert(0, 'evoman')
from evoman.environment import Environment

from evoman_wrapper.EvomanWrapper import EvomanWrapper
from evoman_wrapper.controllers.DemoController import DemoController
from evoman_wrapper.implementations.DemoEvolutionaryAlgorithm import DemoEvolutionaryAlgorithm

"""
1. Set up experiment folder
"""
EXPERIMENT_NAME = "FirstEverExperiment"

experiment_name = EXPERIMENT_NAME
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

"""
2. Setup environment
"""
hidden_layers = 10
environment_config = {
    "experiment_name": EXPERIMENT_NAME,
    "enemies": [2],
    "player_controller": DemoController(n_hidden=hidden_layers),
    "playermode": "ai",
    "enemymode": "static",
    "level": 2,
    "speed": "fastest",
    "logs": "off"
}

environment = Environment(**environment_config)

"""
2. Setup evolutionary algorithm
"""
evolutionary_algorithm_config = {
    "number_of_genomes": (environment.get_num_sensors() + 1) * hidden_layers + (hidden_layers + 1) * 5,
    "number_of_parents": 5,
    "number_of_offsprings": 5,
    "population_size": 10,
    "max_generations": 3,
}
evolutionary_algorithm = DemoEvolutionaryAlgorithm(evolutionary_algorithm_config)

evoman_wrapper = EvomanWrapper(
    environment,
    evolutionary_algorithm
).run_experiment()
