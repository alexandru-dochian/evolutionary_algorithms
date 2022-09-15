from abc import ABC, abstractmethod
import numpy as np
import random


class EvolutionaryAlgorithm(ABC):
    DEFAULT_CONFIG = {
        "number_of_genomes": 100,
        "population_size": 20,
        "number_of_parents": 10,
        "number_of_offsprings": 10,
        "max_generations": 5,
        "distribution_inferior_threshold": -1,
        "distribution_superior_threshold": 1,
        "mutation_chance": 0.2,
    }

    """
        Constructor methods
    """

    def __init__(self, config):
        self.__init_config(config)
        self.__init_metrics()
        self.__init_population()

    def __init_config(self, config: dict = None) -> None:
        self.config = EvolutionaryAlgorithm.DEFAULT_CONFIG

        if config is not None:
            for config_key in config:
                self.config[config_key] = config[config_key]

    def __init_metrics(self) -> None:
        self.fitness_mean_history = []
        self.fitness_max_history = []

    def __init_population(self) -> None:
        self.current_generation_number = 0
        self.population = np.random.uniform(
            self.config["distribution_inferior_threshold"],
            self.config["distribution_superior_threshold"],
            (self.config["population_size"], self.config["number_of_genomes"]))
        self.fitness = np.zeros(self.config["population_size"])
        self.best_individual = random.choice(self.population)
        self.offsprings = np.array([])
        self.mutants = np.array([])

    """
        Abstract methods
    """

    @abstractmethod
    def _selection(self):
        """Implementation must set self.parents"""
        pass

    @abstractmethod
    def _crossover(self):
        """Implementation must set self.offsprings"""
        pass

    @abstractmethod
    def _mutation(self):
        """Implementation must set self.mutants"""
        pass

    """
        Public methods
    """

    def update_fitness(self, fitness: np.array):
        self.fitness = np.vectorize(lambda x: 0 if x < 0 else x)(fitness)

    def next_generation(self):
        if self.current_generation_number > self.config["max_generations"]:
            raise Exception("The max_generations limit({}) has been reached!".format(self.config["max_generations"]))

        self._selection()
        self._crossover()
        self._mutation()
        self.__override_previous_population()

    def finished_evolving(self):
        # TODO Change in (plateau_reached || max_generations_exceeded):
        return self.current_generation_number > self.config["max_generations"]

    def get_generation_description(self):
        return str({
            "current_generation_number": self.current_generation_number,
            "best_individual": self.best_individual.tolist(),
            "best_individual_fitness": self.best_individual_fitness,
            "fitness_max_history": self.fitness_max_history,
            "fitness_mean_history": self.fitness_mean_history
        })

    """
        Private
    """

    def __override_previous_population(self):
        self.population = np.concatenate([self.offsprings, self.mutants])
        self.__update_metrics()

    def __update_metrics(self) -> None:
        self.current_generation_number += 1
        self.fitness_max_history.append(np.max(self.fitness))
        self.fitness_mean_history.append(np.mean(self.fitness))
        self.best_individual = self.population[np.argmax(self.fitness)]
        self.best_individual_fitness = self.fitness[np.argmax(self.fitness)]

    """
        Static methods
    """

    @staticmethod
    def roulette_wheel_selection(population: np.array, fitness: np.array, number_of_parents: int) -> np.array:
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            return np.array([random.choice(population)])

        population_enhanced = list(zip(population, fitness))
        selection_probabilities = [pair[1] / fitness_sum for pair in population_enhanced]
        number_of_individuals, _ = population.shape
        return np.array(
            [population[np.random.choice(number_of_individuals, p=selection_probabilities)]
             for _ in range(number_of_parents)]
        )
