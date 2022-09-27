from copy import copy

import numpy as np
import random

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm

#WITH LINEAR RANKING SELECTION PARAMETERISED BY FACTOR s

class EvolutionaryAlgorithm_2(EvolutionaryAlgorithm):
    def __init__(self, config=None):
        super().__init__(config)

    def _selection(self):
        self.parents = super().LR_selection(
            1.9, self.population, self.fitness, self.config["number_of_parents"])

    def _crossover(self):
        number_of_offsprings = self.config["number_of_offsprings"]
        number_of_parents, number_of_chromosomes = self.parents.shape
        self.offsprings = np.zeros((number_of_offsprings, number_of_chromosomes))

        crossover_point1 = int(number_of_parents / 3)
        crossover_point2 = int(number_of_parents / 2)

        for offspring_index in range(number_of_offsprings):
            first_parent = random.choice(self.parents)
            second_parent = random.choice(self.parents)
            third_parent = random.choice(self.parents)
            self.offsprings[offspring_index][0:crossover_point1] = first_parent[0:crossover_point1]
            self.offsprings[offspring_index][crossover_point1:crossover_point2] = second_parent[crossover_point1:crossover_point2]
            self.offsprings[offspring_index][crossover_point2:] = third_parent[crossover_point2:]

    def _mutation(self):
        mutants = []

        for offspring in self.offsprings:
            mutant = copy(offspring)
            index_1 = random.randrange(len(offspring))
            index_2 = random.randrange(len(offspring))
            while(index_2==index_1 and offspring[index_1] != offspring[index_2]):
                index_2 = random.randrange(len(offspring))
            
            temp_variable = offspring[index_1]
            offspring[index_1] = offspring[index_2]
            offspring[index_2] = temp_variable

            mutants.append(mutant)
        
        self.mutants = np.array(mutants)

