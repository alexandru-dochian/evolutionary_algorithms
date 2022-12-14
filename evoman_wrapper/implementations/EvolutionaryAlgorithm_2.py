from copy import copy

import numpy as np
import random

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm

#WITH TOURNAMENT PARENT SELECTION

class EvolutionaryAlgorithm_2(EvolutionaryAlgorithm):
    def __init__(self, config=None):
        super().__init__(config)

    def _selection(self):
        self.parents = super().tournament_selection(
             self.population, self.fitness, self.config["number_of_parents"])


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
    
    #TODO

    
    def _mutation(self):
        mutants = []

        for offspring in self.offsprings:
            mutant = copy(offspring)
            times = 0 
            first_index = 0
            second_index = 0 
            for genome_index in range(mutant.size):
                if np.random.uniform(0, 1) <= self.config["mutation_chance"]:
                    times += 1
                    mutant[genome_index] = offspring[genome_index] + np.random.normal(-1, 1)
                    if mutant[genome_index] > self.config["distribution_superior_threshold"]:
                        mutant[genome_index] = self.config["distribution_superior_threshold"]
                    elif mutant[genome_index] < self.config["distribution_inferior_threshold"]:
                        mutant[genome_index] = self.config["distribution_inferior_threshold"]
                if times == 1:
                    first_index = genome_index
                elif times == 2:
                    second_index = genome_index
                    index = mutant[first_index]
                    mutant[first_index] = mutant[second_index]
                    mutant[second_index] = index
                    break

            mutants.append(mutant)

        self.mutants = np.array(mutants)

