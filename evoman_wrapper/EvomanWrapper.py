import numpy as np
import json
import os

from evoman_wrapper.interfaces.EvolutionaryAlgorithm import EvolutionaryAlgorithm


class EvomanWrapper:
    def __init__(self, environment, evolutionary_algorithm: EvolutionaryAlgorithm):
        self.environment = environment
        self.evolutionary_algorithm = evolutionary_algorithm

    def run_experiment(self):
        while not self.evolutionary_algorithm.finished_evolving():
            self.__log_generation()
            fitness = self.__compute_fitness(self.evolutionary_algorithm.population)
            self.evolutionary_algorithm.update_fitness(fitness)
            self.evolutionary_algorithm.next_generation()

    def __compute_fitness(self, population: np.array):
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

    def __log_simulation(self, individual_index, simulation_fitness, player_energy, enemy_energy, time_lapsed):
        simulation_log = {
            "current_generation_number": self.evolutionary_algorithm.current_generation_number,
            "Individual Index": individual_index,
            "Fitness": simulation_fitness,
            "Player left energy": player_energy,
            "Enemy left energy": enemy_energy,
            "Time Lapsed": time_lapsed
        }

        self.__persist_and_print(simulation_log, file_name="Individual_{}.json".format(individual_index))

    def __log_generation(self):
        generation_log = self.evolutionary_algorithm.get_generation_description()
        self.__persist_and_print(generation_log, file_name="Overview.json")

    def __persist_and_print(self, log, file_name):
        self.__create_generation_folder()
        final_file_path = "{}/Generation{}/{}".format(
            self.environment.experiment_name, self.evolutionary_algorithm.current_generation_number, file_name)

        with open(final_file_path, 'w') as f:
            json.dump(log, f)
            print("{}\n{}".format(final_file_path, log))

    def __create_generation_folder(self):
        folder_name = "{}/Generation{}".format(
            self.environment.experiment_name, self.evolutionary_algorithm.current_generation_number)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    @staticmethod
    def simulate(environment, individual):
        return environment.play(pcont=individual)
