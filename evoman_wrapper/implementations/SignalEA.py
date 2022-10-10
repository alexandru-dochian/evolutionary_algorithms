import numpy as np
import random

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from evoman_wrapper.utils.ComputationUtils import ComputationUtils

class SignalEA(EvolutionaryAlgorithm):
    MUTATION_SIGNAL = {
        "number_of_sine_functions": 10,
        "min_frequency": 12,
        "max_frequency": 35,
        "min_amplitude": -0.3,
        "amplitude_range": 0.6
    }

    CROSSOVER_SIGNAL = {
            "number_of_sine_functions": 3,
            "min_frequency": 0,
            "max_frequency": 10,
            "min_amplitude": -1,
            "amplitude_range": 2
    }
    
    def __init__(self, config=None):
        super().__init__(config)

    def _selection(self):
        self.parents = super().roulette_wheel_selection(
             self.population, self.fitness, self.config["number_of_parents"])


    def _crossover(self):
        number_of_offsprings = self.config["number_of_offsprings"]
        self.offsprings = np.zeros((number_of_offsprings, self.config["number_of_genomes"]))

        for offspring_index in range(number_of_offsprings):
            mother = random.choice(self.parents)
            father = random.choice(self.parents)
            
            crossover_signal = ComputationUtils.generate_signal(
                SignalEA.CROSSOVER_SIGNAL,
                self.config["number_of_genomes"]
            )
            
            for genome_index in range(crossover_signal.size):
                if crossover_signal[genome_index] > 0:
                    self.offsprings[offspring_index][genome_index] = mother[genome_index]
                else:
                    self.offsprings[offspring_index][genome_index] = father[genome_index]

    def _mutation(self):
        mutants = []

        for individual in np.concatenate([self.parents, self.offsprings]):
             if np.random.uniform(0, 1) <= self.config["mutation_chance"]:
                mutation_signal = ComputationUtils.generate_signal(
                    SignalEA.MUTATION_SIGNAL,
                    self.config["number_of_genomes"]
                )

                mutant = np.multiply(individual, mutation_signal)
                inferior_threshold = self.config["distribution_inferior_threshold"]
                superior_threshold = self.config["distribution_superior_threshold"]
                mutant_limited_to_boundaries = \
                    np.vectorize(lambda genome_value: \
                        EvolutionaryAlgorithm.limit_genome(genome_value, inferior_threshold, superior_threshold)
                    )(mutant)
                mutants.append(mutant_limited_to_boundaries)
        
        self.mutants = np.array(mutants)
