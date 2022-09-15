import sys, os
import numpy as np
from evoman_wrapper.controllers.DemoController import DemoController

sys.path.insert(0, 'evoman')
from evoman.environment import Environment

class Train:

    def run(evolutionary_algorithm):
        experiment_name = 'controller_specialist_demo'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # Update the number of neurons for this specific example
        n_hidden_neurons = 0

        # initializes environment for single objective mode (specialist)  with static enemy and ai player
        env = Environment(experiment_name=experiment_name,
                          playermode="ai",
                          player_controller=DemoController(n_hidden_neurons),
                          speed="normal",
                          enemymode="static",
                          level=2)

        # tests saved demo solutions for each enemy
        for en in range(1, 9):
            # Update the enemy
            env.update_parameter('enemies', [en])

            # Load specialist controller
            sol = np.loadtxt('solutions_demo/demo_' + str(en) + '.txt')
            print("sol", sol)
            print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(en) + ' \n')
            env.play(sol)

