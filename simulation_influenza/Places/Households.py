from .Places import Places

class Households(Places):

    def __init__(self, dict_place_id, 
                strains_keys, vfunc_b_r, lmbd):

        super().__init__(dict_place_id, 
                strains_keys, vfunc_b_r, lmbd)

        self.place = 'sp_hh_id'
        self.Type = 'household'
        self.ages = [0, 100]
