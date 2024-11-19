from .Places import Places
import numpy as np

class Schools(Places):

    def __init__(self, dict_place_id, dict_place_len,
                strains_keys, vfunc_b_r, lmbd):

        super().__init__(dict_place_id, 
                        strains_keys, vfunc_b_r, lmbd)

        self.dict_place_len = dict_place_len
        self.place = 'work_id'
        self.Type = 'school'
        self.ages = [7, 17]

    def prob(self, temp, place_len, x_rand):
        ''' вероятность заражения школьников '''

        # np.random.seed(1)
        length = self.length
        prob_cont = 8.5 / (length - 1) if (8.5 + 1) < length else 1
        res = np.prod(1 - prob_cont * self.lmbd * temp)
        return np.random.binomial(length - 1, 1 - res)
