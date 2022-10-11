import sys, os
from datetime import datetime
import pathlib
import numpy as np
from pathlib import Path

sys.path.insert(0, 'evoman')
from environment import Environment
from evoman_wrapper.persistence.DiskUtils import DiskUtils
from evoman_wrapper.utils.ClassDefinitionLoader import ClassDefinitionLoader


class EvomanWrapper:
    """
        Constructor methods
    """

    def __init__(self, config):
        self.__init_experiment(config)
        self.__init_controller(config["controller"])
        self.__init_environment(config["environment_config"])
        self.__init_evolutionary_algorithm(config["evolutionary_algorithm"])

    def __init_experiment(self, config):
        try:
            # Create experiment directories
            Path("{}/train/".format(config["experiment_name"])).mkdir(parents=True)
            Path("{}/test/".format(config["experiment_name"])).mkdir(parents=True)
        except FileExistsError:
            pass

        self.config = config
        self.__log_config(config["config_type"], config)

    def __init_controller(self, config):
        controller_class_definition = ClassDefinitionLoader.get_controller_instance(config["instance"])
        self.controller = controller_class_definition(**config["constructor"])

    def __init_environment(self, config):
        config["player_controller"] = self.controller
        self.environment = Environment(**config)

    def __init_evolutionary_algorithm(self, config):
        if config:
            evolutionary_algorithm_class_definition = \
                ClassDefinitionLoader.get_evolutionary_algorithm_instance(config["instance"])
            self.evolutionary_algorithm = evolutionary_algorithm_class_definition(config["constructor"])

    """
        Public methods
    """

    def train(self):
        self.__load_state()
        while not self.evolutionary_algorithm.finished_evolving():
            fitness = self.__compute_fitness(self.evolutionary_algorithm.population)
            self.evolutionary_algorithm.update_fitness(fitness)
            self.evolutionary_algorithm.next_generation()
            self.__log_generation()
            self.__dump_state()

    def test(self):
        test_result = {"games": [], "chosen_individual_overview": None}
        chosen_individual_overview = self.__get_best_individual_overview()
        chosen_individual = np.array(chosen_individual_overview["best_individual"])
        for game in self.config["games"]:
            game_log = self.__run_game(game, chosen_individual)
            test_result["games"].append(game_log)
            print("{}\n".format(game_log))

        test_result["chosen_individual_overview"] = chosen_individual_overview
        self.__log_test_result(test_result)

    """
        Private methods
    """

    def __run_game(self, game, best_individual):
        self.environment.update_parameter('enemies', game["enemies"])
        simulation_fitness, player_energy, enemy_energy, time_lapsed = \
            EvomanWrapper.simulate(self.environment, best_individual)

        game["simulation_fitness"] = simulation_fitness
        game["player_energy"] = player_energy
        game["enemy_energy"] = enemy_energy
        game["time_lapsed"] = time_lapsed

        return game

    def __get_best_individual_overview(self):
        path = pathlib.Path("{}/train/Generation{}/Overview.json".format(
            self.config["experiment_name"], self.config["chosen_generation"])).resolve()
        return DiskUtils.load(path)

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
        folder_name = "{}/train/Generation{}".format(
            self.environment.experiment_name, self.evolutionary_algorithm.current_generation_number)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def __dump_state(self):
        state = self.evolutionary_algorithm.dump_state()
        
        final_file_path = "{}/train/dump.json".format(self.environment.experiment_name)
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), state)

    def __load_state(self):
        state = None
        try:
            final_file_path = "{}/train/dump.json".format(self.environment.experiment_name)
            state = DiskUtils.load(pathlib.Path(final_file_path).resolve())
            self.__log("Starting with population from latest dumped generation!")
        except FileNotFoundError:
            self.__log("Initializing population from first generation!")

        self.evolutionary_algorithm.load_state(state)     

    def __log(self, message):
        time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print("{} LOG> {}".format(time, message))

    def __log_simulation(self, individual_index, simulation_fitness, player_energy, enemy_energy, time_lapsed):
        simulation_log = {
            "current_generation_number": self.evolutionary_algorithm.current_generation_number,
            "individual_index": individual_index,
            "simulation_fitness": simulation_fitness,
            "player_energy": player_energy,
            "enemy_energy": enemy_energy,
            "time_lapsed": time_lapsed
        }

        final_file_path = "{}/train/Generation{}/{}".format(
            self.environment.experiment_name,
            self.evolutionary_algorithm.current_generation_number,
            "Individual_{}.json".format(individual_index))
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), simulation_log)

    def __log_test_result(self, test_result):
        final_file_path = "{}/test/result.json".format(self.environment.experiment_name)
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), test_result)

    def __log_generation(self):
        generation_log = self.evolutionary_algorithm.get_generation_description()
        final_file_path = "{}/train/Generation{}/{}".format(
            self.environment.experiment_name,
            self.evolutionary_algorithm.current_generation_number - 1,
            "Overview.json")
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), generation_log)

    @staticmethod
    def __log_config(config_type, config):
        final_file_path = "{}/{}/config.json".format(config["experiment_name"], config_type)
        DiskUtils.store(pathlib.Path(final_file_path).resolve(), config)

    @staticmethod
    def simulate(environment, individual):
        return environment.play(pcont=individual)
