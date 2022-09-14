from abc import ABC, abstractmethod
import numpy as np
import random
import typing


class EvolutionaryAlgorithm(ABC):
    DEFAULT_CONFIG = {
        "number_of_parents": 20,
        "number_of_offsprings": 20,
        "max_generations": 5,
        "first_generation_index": 0,
    }

    """
        Constructor methods
    """

    def __init__(self, config):
        self.__init_config(config)
        self.__init_metrics()

    def __init_config(self, config: dict) -> None:
        if config is None:
            self.config = EvolutionaryAlgorithm.DEFAULT_CONFIG
        else:
            self.config = config

    def __init_metrics(self) -> None:
        self.fitness_mean_history = []
        self.fitness_max_history = []

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

        self.__selection()
        self.__crossover()
        self.__mutation()
        self.__override_previous_generation()

    """
        Override previous generation
    """

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

    def __roulette_wheel_selection(self) -> np.array:
        fitness_sum = self.fitness.sum()
        if fitness_sum == 0:
            return np.array([random.choice(self.generation)])

        generation_enhanced = list(zip(self.generation, self.fitness))
        selection_probabilities = [pair[1] / fitness_sum for pair in generation_enhanced]
        number_of_individuals, _ = self.generation.shape
        return np.array(
            [self.generation[np.random.choice(number_of_individuals, p=selection_probabilities)]
             for _ in range(self.config["number_of_parents"])]
        )

    """
        Abstract methods
    """

    @abstractmethod
    def __selection(self):
        pass

    @abstractmethod
    def __crossover(self):
        pass

    @abstractmethod
    def __mutation(self):
        pass
