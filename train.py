from evoman_wrapper.Train import Train
from evoman_wrapper.implementations.DemoEvolutionaryAlgorithm import DemoEvolutionaryAlgorithm

config = {
        "number_of_parents": 100,
        "number_of_offsprings": 69,
        "max_generations": 5,
}

evolutionary_algorithm = DemoEvolutionaryAlgorithm(config)

Train.run(evolutionary_algorithm)
