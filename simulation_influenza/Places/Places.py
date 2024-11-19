import numpy as np
import random

class Places:

    def __init__(self, dict_place_id, 
                strains_keys, vfunc_b_r, lmbd):

        self.dict_place_id = dict_place_id
        self.strains_keys = strains_keys
        self.vfunc_b_r = vfunc_b_r
        self.lmbd = lmbd

        self.dict_place_len = None
        self.place = None
        self.Type = None
        self.ages = None
        

    def prob(self, temp, place_len, x_rand):
        ''' вероятность заражения людей '''

        # np.random.seed(1)
        # Вер. заражения (lambda*g(tau))
        prob = np.repeat(temp, place_len) * self.lmbd
        curr_length = len(prob)
        place_rand = x_rand[:curr_length]


        np.roll(x_rand, curr_length)

        return len(place_rand[place_rand < prob])
 

    def preprocess_for_workplaces(self, data_current):
        ''' выбор рабочих для процесса заражения на рабочих местах '''
        ...

    def infect(self, place_inf_dic, data_current, x_rand, j):
        ''' процесс заражения в одном из типов мест за 1 день '''

        # np.random.seed(1)
        real_inf_place_dic = {}

        for key in self.strains_keys:
            real_inf_place_dic[key] = np.array([])


        self.preprocess_for_workplaces(data_current)


        place_inf_dic_keys_shuffled = list(place_inf_dic.keys())
        
        # np.random.seed(1)
        # random.seed(1)
        random.shuffle(place_inf_dic_keys_shuffled)

        # Все места, где есть инфицированные любым из штаммов
        for place_id, cur_strain in place_inf_dic_keys_shuffled:

            # Нет данных о жителях в словаре мест по id для данного штамма
            if place_id not in self.dict_place_id[cur_strain].keys():
                    # TODO: убрать сортировку после дебага
                    self.dict_place_id[cur_strain].update(
                    {place_id: sorted(list(data_current[
                        (data_current[self.place] ==place_id) &
                        (data_current['susceptible_'+cur_strain]==1) &
                        (data_current.age>=self.ages[0]) &
                        (data_current.age<=self.ages[1])
                    ].index))})

            place_len = len(self.dict_place_id[cur_strain][place_id])

            if self.Type == 'school':
                self.length = self.dict_place_len[place_id]

            if place_len != 0:
                # Инфекционность для всех больных c данным штаммом
                temp = self.vfunc_b_r(place_inf_dic[(place_id, cur_strain)])
                real_inf = self.prob(temp, place_len, x_rand)

                if place_len < real_inf:
                    real_inf = place_len

                # Выбираем, кто заразился, из восприимчивых
                # np.random.seed(1)
                real_inf_id = np.random.choice(
                    np.array(self.dict_place_id[cur_strain][place_id]), real_inf, replace=False)

                # Дописываем инфицированных по штаммам и домохозяйствам
                real_inf_place_dic[cur_strain] = np.concatenate(
                    (real_inf_place_dic[cur_strain], real_inf_id))
                
        return real_inf_place_dic
