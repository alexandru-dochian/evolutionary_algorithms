import numpy as np
import random

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm


class DemoEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def __init__(self, config=None):
        super().__init__(config)

    @staticmethod
    def __individual_fitness(individual: np.array) -> np.array:
        # TODO: implement individual fitness
        return random.randint(0, 100) if individual else random.randint(0, 100)

    def _fitness(self) -> np.array:
        self.fitness = np.array([self.__individual_fitness(individual) for individual in self.generation])

    def _selection(self):
        self.parents = super().roulette_wheel_selection()

    def _crossover(self):
        number_of_offsprings = self.config["number_of_offsprings"]
        number_of_parents, number_of_chromosomes = self.parents.shape
        self.offsprings = np.zeros((number_of_offsprings, number_of_chromosomes))

        crossover_point = int(number_of_parents / 2)

        for offspring_index in range(number_of_offsprings):
            first_parent = random.choice(self.parents)
            second_parent = random.choice(self.parents)
            self.offsprings[offspring_index][0:crossover_point] = first_parent[0:crossover_point]
            self.offsprings[offspring_index][crossover_point:] = second_parent[crossover_point:]

    def _mutation(self):
        self.mutants = self.offsprings
