from .Households import Households
from .Workplaces import Workplaces
from .Schools import Schools

class PlaceFactory:

    def __init__(self, strains_keys, vfunc_b_r, lmbd):
        ''' фабрика создания обьектов классов мест, где происходит заражение '''
        self.strains_keys = strains_keys
        self.vfunc_b_r = vfunc_b_r
        self.lmbd = lmbd

    def Households(self, dict_place_id):
        ''' создание обьекта класса домовладения '''
        return Households(
                    dict_place_id, 
                    self.strains_keys, 
                    self.vfunc_b_r, 
                    self.lmbd
                )

    def Workplaces(self, dict_place_id):
        ''' создание обьекта класса рабочего места '''
        return Workplaces(
                    dict_place_id, 
                    self.strains_keys, 
                    self.vfunc_b_r, 
                    self.lmbd
                )

    def Schools(self, dict_place_id, dict_place_len):
        ''' создание обьекта класса школы '''
        return Schools(
                    dict_place_id,
                    dict_place_len,
                    self.strains_keys,
                    self.vfunc_b_r,
                    self.lmbd
                )
