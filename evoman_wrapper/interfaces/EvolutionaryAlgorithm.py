from abc import ABC, abstractmethod
import numpy as np
import random
import typing


class EvolutionaryAlgorithm(ABC):
    DEFAULT_CONFIG = {
        "number_of_parents": 20,
        "number_of_offsprings": 20,
        "max_generations": 5,
    }

    """
        Constructor methods
    """

    def __init__(self, config):
        self.__init_config(config)
        self.__init_metrics()

    def __init_config(self, config: dict = None) -> None:
        self.config = EvolutionaryAlgorithm.DEFAULT_CONFIG

        if config is not None:
            for config_key in config:
                self.config[config_key] = config[config_key]

    def __init_metrics(self) -> None:
        self.fitness_mean_history = []
        self.fitness_max_history = []

    """
        Abstract methods
    """
    @abstractmethod
    def _fitness(self):
        """Implementation must set self.fitness"""
        pass

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
    def compute_and_log_new_generation(self) -> typing.Generator:
        while abs(100 - self.accuracy_percentage) > self.config["error"]:
            self.__compute_new_generation()
            yield self.__log_current_generation()

    def __compute_new_generation(self) -> None:
        if self.current_generation_number > self.config["max_generations"]:
            raise Exception("The max_generations limit({}) has been reached!".format(self.config["max_generations"]))

        self._selection()
        self._crossover()
        self._mutation()
        self.__override_previous_generation()

    def __override_previous_generation(self):
        self.generation = np.concatenate([self.offsprings, self.mutants])
        self.fitness = self.__compute_fitness()
        self.__update_best_individual()
        self.__update_metrics()

    def __log_current_generation(self) -> None:
        # TODO: store logs in json format to be used in graphs
        raise NotImplemented()

    def __update_metrics(self) -> None:
        self.fitness_max_history.append(np.max(self.fitness))
        self.fitness_mean_history.append(np.mean(self.fitness))


    """
        Static methods
    """

    @staticmethod
    def roulette_wheel_selection(generation: np.array, fitness: np.array, number_of_parents: int) -> np.array:
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            return np.array([random.choice(generation)])

        generation_enhanced = list(zip(generation, fitness))
        selection_probabilities = [pair[1] / fitness_sum for pair in generation_enhanced]
        number_of_individuals, _ = generation.shape
        return np.array(
            [generation[np.random.choice(number_of_individuals, p=selection_probabilities)]
             for _ in range(number_of_parents)]
        )
