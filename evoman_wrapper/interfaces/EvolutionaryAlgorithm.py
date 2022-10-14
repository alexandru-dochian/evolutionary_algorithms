from abc import ABC, abstractmethod
from contextlib import nullcontext
import numpy as np
import random

from evoman_wrapper.utils.ComputationUtils import ComputationUtils

class EvolutionaryAlgorithm(ABC):
    DEFAULT_CONFIG = {
        "number_of_genomes": 100,
        "number_of_parents": 10,
        "number_of_offsprings": 10,
        "max_generations": 5,
        "distribution_inferior_threshold": -1,
        "distribution_superior_threshold": 1,
        "mutation_chance": 0.5,
    }

    """
        Constructor methods
    """

    def __init__(self, config):
        self.__init_config(config)
        self.__init_metrics()

    def __init_config(self, config: dict = None) -> None:
        self.config = EvolutionaryAlgorithm.DEFAULT_CONFIG

        if config is not None:
            for config_key in config:
                self.config[config_key] = config[config_key]

    def __init_metrics(self) -> None:
        self.fitness_mean_history = []
        self.fitness_max_history = []
        self.fitness_min_history = []

    """
        Abstract methods
    """

    @abstractmethod
    def _selection(self):
        """Implementation must set self.parents"""
        pass

    @abstractmethod
    def _crossover(self):
        """Implementation must set self.offsprings"""
        pass

    @abstractmethod
    def _mutation(self):
        """Implementation must set self.mutants"""
        pass

    """
        Public methods
    """

    def update_fitness(self, fitness: np.array):
        self.fitness = fitness

    def next_generation(self):
        if self.current_generation_number > self.config["max_generations"]:
            raise Exception("The max_generations limit({}) has been reached!".format(self.config["max_generations"]))

        self._selection()
        self._crossover()
        self._mutation()
        self.__override_previous_population()

    def finished_evolving(self):
        # TODO Change in (plateau_reached || max_generations_exceeded):
        return self.current_generation_number > self.config["max_generations"]

    def get_generation_description(self):
        return {
            "current_generation_number": self.current_generation_number - 1,
            "best_individual": self.best_individual.tolist(),
            "best_individual_fitness": self.best_individual_fitness,
            "fitness_max_history": self.fitness_max_history,
            "fitness_mean_history": self.fitness_mean_history
        }

    def dump_state(self):
        return {
            "population": self.population,
            "current_generation_number": self.current_generation_number
        }

    def load_state(self, state = None):
        if state is None:
            self.current_generation_number = 0
            self.population = np.random.uniform(
                self.config["distribution_inferior_threshold"],
                self.config["distribution_superior_threshold"],
                (self.config["number_of_parents"] + self.config["number_of_offsprings"], self.config["number_of_genomes"]))
        else:
            self.population = state["population"]
            self.current_generation_number = state["current_generation_number"]

        self.fitness = np.zeros(self.population.size)
        self.best_individual = random.choice(self.population)
        self.best_individual_fitness = 0
        self.offsprings = np.array([])
        self.mutants = np.array([])
    
    """
        Private
    """

    def __override_previous_population(self):
        self.__update_metrics()
        self.population = np.concatenate([self.parents, self.offsprings, self.mutants])

    def __update_metrics(self) -> None:
        self.current_generation_number += 1
        self.fitness_max_history.append(np.max(self.fitness))
        self.fitness_mean_history.append(np.mean(self.fitness))
        self.fitness_min_history.append(np.min(self.fitness))
        self.best_individual = self.population[np.argmax(self.fitness)]
        self.best_individual_fitness = self.fitness[np.argmax(self.fitness)]

    """
        Static methods
    """

    @staticmethod
    def roulette_wheel_selection(population: np.array, fitness: np.array, number_of_parents: int) -> np.array:
        fitness = np.vectorize(lambda x: ComputationUtils.norm(x, fitness))(fitness)
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            return np.array([random.choice(population)])

        population_enhanced = list(zip(population, fitness))
        selection_probabilities = [pair[1] / fitness_sum for pair in population_enhanced]
        number_of_individuals, _ = population.shape
        return np.array(
            [population[np.random.choice(number_of_individuals, p=selection_probabilities)]
             for _ in range(number_of_parents)]
        )

    
    @staticmethod
    def tournament_selection(population: np.array, fitness: np.array, number_of_parents: int) -> np.array:
        population_size = len(population)
        firstind = random.randint(0, population_size-1)
        firstfitness = fitness[firstind]
        secondfitness = -1000
        secondind = 0
        for i in range(1, 9):
            ind = random.randint(0,population_size-1)
            if fitness[ind] > firstfitness:
                tempfitness = firstfitness
                tempind = firstind
                firstind = ind
                firstfitness = fitness[ind]
                if secondfitness > -1000:
                    secondfitness = tempfitness
                    secondind = tempind
            elif fitness[ind] > secondfitness:
                secondfitness = fitness[ind]
                secondind = ind
        return population[firstind], population[secondind]



    @staticmethod
    #Rank-based Selection: Linear Ranking parameterised by factor s
    def LR_selection(population: np.array, fitness: np.array, number_of_parents: int) -> np.array:
        
        fitness = np.vectorize(lambda x: ComputationUtils.norm(x, fitness))(fitness)
        fitness_sum = fitness.sum()
        if fitness_sum == 0:
            #print("fitness sum = 0")
            return np.array([random.choice(population)])
        
        #Compute ranking
        ranking = [sorted(fitness).index(x) for x in fitness]
        population_enhanced = list(zip(population, ranking))
        population_enhanced = sorted(population_enhanced, key = lambda item: item[1])


        #Get the selection probabilities
        s = 1.9
        population_size = len(population)
        probability_first_part = (2-s) / population_size
        probability_second_part = lambda rank_index : 2 * rank_index * (s-1) / (population_size * (population_size - 1))
        selection_probabilities = np.array([probability_first_part + probability_second_part(rank_index) for rank_index in range(len(ranking))])
            
        

        number_of_individuals, _ = population.shape

        return np.array(
            [population[np.random.choice(number_of_individuals, p = selection_probabilities)]
             for _ in range(number_of_parents)]
        )    


    @staticmethod
    def limit_genome(genome_value , inferior_threshold, superior_threshold):
        if genome_value > superior_threshold:
            return superior_threshold
        elif genome_value < inferior_threshold:
            return inferior_threshold
        else:
            return genome_value
