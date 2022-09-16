import os
import pathlib
import numpy as np

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from evoman_wrapper.persistence.DiskUtils import DiskUtils


class EvomanWrapper:
    def __init__(self, environment, evolutionary_algorithm: EvolutionaryAlgorithm):
        self.environment = environment
        self.evolutionary_algorithm = evolutionary_algorithm

    def run_experiment(self):
        while not self.evolutionary_algorithm.finished_evolving():
            fitness = self.__compute_fitness(self.evolutionary_algorithm.population)
            self.evolutionary_algorithm.update_fitness(fitness)
            self.evolutionary_algorithm.next_generation()
            self.__log_generation()

    def __compute_fitness(self, population: np.array):
        self.__create_generation_folder()

        # TODO: parallelize this
        fitness = np.array([])

        for individual_index in range(0, population.shape[0]):
            individual = population[individual_index]
            simulation_fitness, player_energy, enemy_energy, time_lapsed \
                = EvomanWrapper.simulate(self.environment, individual)
            fitness = np.append(fitness, simulation_fitness)
            self.__log_simulation(
                individual_index, simulation_fitness, player_energy, enemy_energy, time_lapsed)

        return fitness

    def __create_generation_folder(self):
        folder_name = "{}/Generation{}".format(
            self.environment.experiment_name, self.evolutionary_algorithm.current_generation_number)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def __log_simulation(self, individual_index, simulation_fitness, player_energy, enemy_energy, time_lapsed):
        simulation_log = {
            "current_generation_number": self.evolutionary_algorithm.current_generation_number,
            "Individual Index": individual_index,
            "Fitness": simulation_fitness,
            "Player left energy": player_energy,
            "Enemy left energy": enemy_energy,
            "Time Lapsed": time_lapsed
        }

        final_file_path = "{}/Generation{}/{}".format(
            self.environment.experiment_name,
            self.evolutionary_algorithm.current_generation_number,
            "Individual_{}.json".format(individual_index))
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), simulation_log)

    def __log_generation(self):
        generation_log = self.evolutionary_algorithm.get_generation_description()
        final_file_path = "{}/Generation{}/{}".format(
            self.environment.experiment_name,
            self.evolutionary_algorithm.current_generation_number - 1,
            "Overview.json")
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), generation_log)

    @staticmethod
    def simulate(environment, individual):
        return environment.play(pcont=individual)
