import numpy as np
import random

from src.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm


class DemoEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __selection(self):
        pass

    @abstractmethod
    def __crossover(self):
        pass

    @abstractmethod
    def __mutation(self):
        pass




    """
        Selection
    """

    def __selection(self) -> np.array:
        self.parents = super().__roulette_wheel_selection()

    """
        Crossover
    """

    def __crossover(self):
        number_of_offsprings = self.config["number_of_offsprings"]
        number_of_parents, number_of_chromosomes = self.parents.shape
        offsprings = np.zeros((number_of_offsprings, number_of_chromosomes))

        crossover_point = int(number_of_parents / 2)

        for offspring_index in range(number_of_offsprings):
            first_parent = random.choice(self.parents)
            second_parent = random.choice(self.parents)
            offsprings[offspring_index][0:crossover_point] = first_parent[0:crossover_point]
            offsprings[offspring_index][crossover_point:] = second_parent[crossover_point:]

        return offsprings

    """
        Mutation
    """

    def __mutation(self):
        # TODO: implement mutation
        self.mutants = self.offsprings

    """
        Fitness
    """

    def __compute_fitness(self) -> np.array:
        return np.array([self.__individual_fitness(individual) for individual in self.generation])

    def __individual_fitness(self, individual: np.array) -> np.array:
        # TODO: implement individual fitness
        if individual:

            return random.randint(0, 100)
        else:
            return random.randint(0, 100)

    def __update_metrics(self) -> None:
        self.fitness_max_history.append(np.max(self.fitness))
        self.fitness_mean_history.append(np.mean(self.fitness))
        self.current_generation_number += 1

    def __get_best_individual(self) -> np.array:
        return self.generation[np.argmax(self.fitness)]
