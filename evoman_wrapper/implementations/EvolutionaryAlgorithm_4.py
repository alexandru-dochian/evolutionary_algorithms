import numpy as np
import random

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from evoman_wrapper.utils.ComputationUtils import ComputationUtils

class EvolutionaryAlgorithm_4(EvolutionaryAlgorithm):
    def __init__(self, config=None):
        super().__init__(config)

    def _selection(self):
        mother, father = super().tournament_selection(self.population, self.fitness, self.config["number_of_parents"])

    
    #def _selection(self):
        #self.parents = super().LR_selection(
             #self.population, self.fitness, self.config["number_of_parents"])    

    def _crossover(self):
        number_of_offsprings = self.config["number_of_offsprings"]
        self.offsprings = np.zeros((number_of_offsprings, self.config["number_of_genomes"]))
        self.parents = []
        for offspring_index in range(number_of_offsprings):
            mother, father = super().tournament_selection(self.population, self.fitness, self.config["number_of_parents"])
            self.parents.append(mother)
            self.parents.append(father)

            crossover_signal = ComputationUtils.generate_signal(
                self.config["crossover_signal_config"],
                self.config["number_of_genomes"]
            )

            for genome_index in range(crossover_signal.size):
                if crossover_signal[genome_index] > 0:
                    self.offsprings[offspring_index][genome_index] = mother[genome_index]
                else:
                    self.offsprings[offspring_index][genome_index] = father[genome_index]

    #def _crossover(self):
        #number_of_offsprings = self.config["number_of_offsprings"]
        #self.offsprings = np.zeros((number_of_offsprings, self.config["number_of_genomes"]))

        #for offspring_index in range(number_of_offsprings):
            #mother = random.choice(self.parents)
            #father = random.choice(self.parents)
            
            #crossover_signal = ComputationUtils.generate_signal(
                #self.config["crossover_signal_config"],
                #self.config["number_of_genomes"]
            #)
            
            #for genome_index in range(crossover_signal.size):
                #if crossover_signal[genome_index] > 0:
                    #self.offsprings[offspring_index][genome_index] = mother[genome_index]
                #else:
                    #self.offsprings[offspring_index][genome_index] = father[genome_index]

    def _mutation(self):
        mutants = []

        individual_pool = np.concatenate([self.parents, self.offsprings])
        for _ in range(self.config["number_of_mutants"]):
             if np.random.uniform(0, 1) <= self.config["mutation_chance"]:
                mutation_signal = ComputationUtils.generate_signal(
                    self.config["mutation_signal_config"],
                    self.config["number_of_genomes"]
                )

                individual = random.choice(individual_pool)
                mutant = individual + mutation_signal
                #mutant = np.multiply(individual, mutation_signal)
                inferior_threshold = self.config["distribution_inferior_threshold"]
                superior_threshold = self.config["distribution_superior_threshold"]
                mutant_limited_to_boundaries = \
                    np.vectorize(lambda genome_value: \
                        EvolutionaryAlgorithm.limit_genome(genome_value, inferior_threshold, superior_threshold)
                    )(mutant)
                mutants.append(mutant_limited_to_boundaries)
        
        self.mutants = np.array(mutants)

