import numpy as np
import random

class ComputationUtils:
    @staticmethod
    def norm(value, array: np.array):
        if (array.max() - array.min()) > 0:
            norm = ( value - array.min() )/( array.max() - array.min() )
        else:
            norm = 0

        if norm <= 0:
            return 1e-10
        
        return norm


    @staticmethod
    def generate_signal(signal_config: dict, number_of_points: int):
        sampling_frequency = 1000
        time_array = np.arange(number_of_points)
        signal = np.zeros(number_of_points)

        frequency_range = signal_config["max_frequency"] - signal_config["min_frequency"]
        for _ in range(signal_config["number_of_sine_functions"]):
            random_starting_phase = random.random() * 2 * np.pi
            signal_frequency = int(random.random() * frequency_range + signal_config["min_frequency"])
            random_signal = np.sin(random_starting_phase + signal_frequency / sampling_frequency * time_array)

            random_signal = np.vectorize(lambda x: ComputationUtils.norm(x, random_signal))(random_signal)
            random_signal = (random_signal * signal_config["amplitude_range"]) + signal_config["min_amplitude"]
            signal += random_signal

        return signal
