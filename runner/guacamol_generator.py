from tqdm import tqdm
from joblib import Parallel

from copy import deepcopy

import numpy as np
import torch

from guacamol.goal_directed_generator import GoalDirectedGenerator

record_w_neptune = False
if record_w_neptune:
    import neptune


class GeneticExpertGuidedLearningGenerator(GoalDirectedGenerator):
    def __init__(self, trainer, recorder, num_steps, device, scoring_num_list, num_jobs, tracker):
        self.trainer = trainer
        self.recorder = recorder
        self.num_steps = num_steps
        self.device = device
        self.scoring_num_list = scoring_num_list

        self.best_score_seen = 0
        self.tracker = tracker

        self.pool = Parallel(n_jobs=num_jobs)

    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population):
        self.trainer.init(scoring_function=scoring_function, device=self.device, pool=self.pool)
        for step in tqdm(range(self.num_steps)):
            smis, scores = self.trainer.step(
                scoring_function=scoring_function, device=self.device, pool=self.pool
            )
            # import pdb
            # pdb.set_trace()
            max_new_score = max(scores)
            self.best_score_seen = max(self.best_score_seen, max_new_score)
            print("Num function evals so far:", self.trainer.total_num_evals) # num new evals == len(scores) obviously 
            print("best score seen:", self.best_score_seen)
            self.tracker.log({"num_func_evals:":self.trainer.total_num_evals, "best_score_seen":self.best_score_seen })
            
            self.recorder.add_list(smis=smis, scores=scores)
            self.recorder.log()

        self.recorder.log_final()

        best_smis, _ = self.recorder.get_topk(k=number_molecules)

        return best_smis
