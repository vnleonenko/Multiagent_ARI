from .Places import Places
import numpy as np

class Workplaces(Places):

    def __init__(self, dict_place_id, 
                strains_keys, vfunc_b_r, lmbd):

        super().__init__(dict_place_id, 
                strains_keys, vfunc_b_r, lmbd)

        self.place = 'work_id'
        self.Type = 'workplace'
        self.ages = [18, 100]

    def preprocess_for_workplaces(self, data_current):
        ''' выбор рабочих для процесса заражения на рабочих местах '''

        # np.random.seed(1)
        data_current = data_current[(data_current.work_id != 0) & (
            data_current.age > 17)]  # Подможество работающих из всей популяции

        return True
