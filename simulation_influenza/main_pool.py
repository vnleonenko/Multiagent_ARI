import numpy as np
from collections import defaultdict
from functools import partial
import datetime
import pandas as pd
import json
import multiprocessing as mp
import os
import warnings

from simulation_influenza.tools import Preprocess, file_timer, day_timer
from simulation_influenza.Places import PlaceFactory

warnings.filterwarnings('ignore')
pd.options.display.width = None
pd.options.display.max_columns = None

def func_b_r(inf_day):
    a = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]
    if inf_day < 9:
        return a[inf_day - 1]
    else:
        return 0

class Main:
    def __init__(self, strains_keys, infected_init, alpha, lmbd):
        ''' создание обьекта типа Main и задание основных данных '''
        self.strains_keys = strains_keys
        self.infected_init_dic = {i: j for i,j in zip(strains_keys, infected_init)}
        self.alpha_dic = {i: j for i,j in zip(strains_keys, alpha)}
        self.lmbd = lmbd

        self.vfunc_b_r = np.vectorize(func_b_r)
        return None


    def runs_params(self, num_runs, days, data_folder):
        ''' задание параметров для запусков и загрузка данных '''
        self.num_runs = num_runs
        self.days = range(days[0], days[1]+1)
        self.data_folder = data_folder

        current_file = os.path.realpath(__file__)
        self.data_path = os.path.dirname(current_file)+ '/../data/' + data_folder + '/'
        self.results_dir = os.path.dirname(current_file) + '/../results/' + data_folder + '/'
        self.make_dir(self.data_path, False)
        self.make_dir(self.results_dir, True)

        return True

    
    def age_groups_params(self, age_groups, vaccined_fraction):
        ''' задание возрастных групп и предобработка данных '''
        self.age_groups = age_groups
        self.vaccined_fraction = vaccined_fraction

        Preprocess(self)

        return True



    @file_timer
    def start(self, with_seirb=False):
        ''' функция запуска вычислений '''
        # np.random.seed(1)

        data = self.data_current
        print(data[data.susceptible_H1N1==0])

        cpu_num = mp.cpu_count()
        print("{} processors detected".format(cpu_num))

        print("Starting from I0: ")
        print(data[data.illness_day > 2].sp_id)

        with mp.Pool(self.num_runs) as pool:
                #pool.map(self.main, range(self.num_runs),
                pool.map(partial(self.main, with_seirb=with_seirb), 
                         range(self.num_runs))

        return True


    def main(self, number_seed, with_seirb=False):
        ''' основной цикл работы '''
        # np.random.seed(1)

        self.place_dict = dict()
        self.results_dic = defaultdict(list)
        self.new_results_dic, self.old_res_dict = defaultdict(list), defaultdict(list)
        self.child_res_dict, self.adult_res_dict = defaultdict(list), defaultdict(list)
        
        # TODO: удалить после дебага
        for key in self.strains_keys:
            for sc in self.dict_school_id[key]:
                self.dict_school_id[key][sc] = sorted(self.dict_school_id[key][sc])

        # TODO: можно вынести в start, чтобы не создавать в каждом потоке отдельно
        #       но будет некравсиво выглядеть
        PF = PlaceFactory(self.strains_keys, self.vfunc_b_r, self.lmbd)

        # создание обьектов мест
        self.H = PF.Households(self.dict_hh_id)
        self.W = PF.Workplaces(self.dict_work_id)
        self.S = PF.Schools(self.dict_school_id, self.dict_school_len)
        
        if with_seirb:
            # dataset со всеми данными на каждый день
            self.cols = ['S','E','I','R','beta']
            fin_cols = [f'{col}_{strain}' for strain in self.strains_keys 
                                            for col in self.cols]
            self.SEIRb_day = pd.DataFrame.from_records(np.zeros((len(self.days),
                                                                 len(fin_cols))),
                                                       columns=fin_cols)
            self.SEIRb_day.index = self.days
            
        # инфецирование по дням
        for j in self.days:
            self.day(j, number_seed, with_seirb)
           

        # выгрузка результатов в файлы
        self.json_from_dict(self.place_dict, r'/inf_people_{}.json', number_seed)
        self.csv_from_dict(self.adult_res_dict, r'/adult_incidence_{}.csv', number_seed)
        self.csv_from_dict(self.child_res_dict, r'/child_incidence_{}.csv', number_seed)
        self.csv_from_dict(self.old_res_dict, r'/old_incidence_{}.csv', number_seed)

        return True  


    @day_timer
    def day(self, j, number_seed, with_seirb=False):
        ''' симуляция 1 дня '''
        if len(self.data_current[self.data_current.illness_day > 2]) != 0:
            x_rand = np.random.rand(1000000)

            # словари с id мест, где есть инфецированные
            hh_inf_dic, work_inf_dic, school_inf_dic = self.current_infected(self.data_current)

            # инфецированные на местах
            real_inf_hh_dic = self.H.infect(hh_inf_dic, self.data_current, x_rand, j)
            real_inf_work_dic = self.W.infect(work_inf_dic, self.data_current, x_rand, j)
            real_inf_school_dic = self.S.infect(school_inf_dic, self.data_current, x_rand, j)
                    
            # сбор id заболевших людей со всех мест
            real_inf_results_dic = {}
            # np.random.seed(1)
            for key in self.strains_keys:  
                real_inf_results_dic[key] = np.concatenate((
                        real_inf_hh_dic[key], 
                        real_inf_school_dic[key], 
                        real_inf_work_dic[key]
                ))
                real_inf_results_dic[key] = np.unique(
                    real_inf_results_dic[key].astype(int)
                )

            # обновление данных для больных
            for key in self.strains_keys:
                self.data_current.loc[
                    real_inf_results_dic[key], 
                    ['infected', 'illness_day', 'susceptible_'+key]
                ] = [self.IndexForStrain(key), 1, 0]
                
            # Обновление словарей c id восприимчивых людей на местах
            for key in self.strains_keys:  
                self.update_dict(key, 'sp_hh_id', self.dict_hh_id, real_inf_results_dic)
                self.update_dict(key, 'work_id', self.dict_work_id, real_inf_results_dic)
                self.update_dict(key, 'work_id', self.dict_school_id, real_inf_results_dic)
                
                # и добавление данных: S, E, I, R, beta
                if with_seirb:
                    strain_index = self.IndexForStrain(key)
                    n_days_exposed = 2 # сколько дней не может заражать

                    S = self.data_current[f'susceptible_{key}'].sum()
                    # т.к. на 1-м и 2-м дне заразить не может (по ф-ии func_b_r)
                    E = self.data_current[(self.data_current.infected == strain_index) & (
                                            self.data_current.illness_day <= n_days_exposed)
                                         ].shape[0]
                    I = self.data_current[(self.data_current.infected == strain_index) & (
                                            self.data_current.illness_day > n_days_exposed)
                                         ].shape[0]
                    R = self.data_current[(self.data_current[f'susceptible_{key}'] == 0) & (
                                            self.data_current.infected == 0)
                                         ].shape[0]

                    new_i = self.data_current[(self.data_current.infected == strain_index) & (
                                              self.data_current.illness_day == n_days_exposed+1)
                                             ].shape[0]

                    beta_value = new_i / (S*I)
                    
                    key_cols = [f'{col}_{key}' for col in self.cols]
                    self.SEIRb_day.loc[j, key_cols] = [S, E, I, R, beta_value]

        self.infected_info(j, number_seed)

        self.csv_from_dict(self.results_dic, r'/prevalence_seed_{}.csv', number_seed)
        self.csv_from_dict(self.new_results_dic, r'/incidence_seed_{}.csv', number_seed)
        if with_seirb:
            self.csv_from_df(self.SEIRb_day, r'/seirb_seed_{}.csv', number_seed)
        
        # отслеживание передачи заболевания в доме
        self.place_dict[j] = list(self.data_current[self.data_current.illness_day>0].sp_hh_id)

        self.update_illness()

        return True


    def infected_info(self,  j, number_seed):
        ''' обновление информации по заболевшим '''
        # np.random.seed(1)

        for key in self.strains_keys:
                
            newly = self.data_current[
                (self.data_current.illness_day == 1) & 
                (self.data_current.infected == self.IndexForStrain(key))
            ]

            newly_infected_by_strain = len(newly)

            newly_old = len(newly[newly.age>59])
            newly_child = len(newly[newly.age<18])
            newly_adult = len(newly[(newly.age>=18)&(newly.age<=59)])

            infected_by_strain = len(self.data_current[
                    self.data_current.infected == self.IndexForStrain(key)
            ])

            self.results_dic[key].append(infected_by_strain)
            self.new_results_dic[key].append(newly_infected_by_strain)
            self.old_res_dict[key].append(newly_old)
            self.child_res_dict[key].append(newly_child)
            self.adult_res_dict[key].append(newly_adult)

            dataset_infected_by_strain = self.data_current[
                self.data_current.infected == self.IndexForStrain(key)
            ][['sp_id', 'sp_hh_id']]

            dataset_infected_by_strain = dataset_infected_by_strain.sort_values(by=[
                                                                                'sp_hh_id'])
            c_inf = int(self.data_current[
                self.data_current.infected ==self.IndexForStrain(key)
            ]['infected'].sum())
        

            print("{}: day {} {} I {}({}) S {} Time {}".format(number_seed, j, key, newly_infected_by_strain, c_inf,
                    int(self.data_current[['susceptible_'+key]].sum()), datetime.datetime.now()))

        return True


    def update_illness(self):
        ''' обновление информации по больным '''
        
        # обновляем день болезни у людей
        self.data_current.loc[
                self.data_current.infected > 0, 
                'illness_day'
        ] += 1 

        # добавление имунитета переболевшим
        for key in self.strains_keys:
            self.data_current.loc[
                    self.data_current.illness_day > 8, 
                    ['susceptible_'+key]
            ] = 0

        # выздоравление
        self.data_current.loc[
                self.data_current.illness_day > 8, 
                ['infected', 'illness_day']
            ] = 0
        
        return True



    def update_dict(self, key, type, dict_id, real_inf_results_dic):
        ''' обновление информации в словарях с восприимчивыми в местах '''

        current_id = []
        [current_id.extend(i) for i in dict_id[key].values()]

        check_id = [
            True if i in current_id 
            else False 
            for i in real_inf_results_dic[key]
        ]
        
        check_id = [i for i, x in enumerate(check_id) if x is True]
        inf = real_inf_results_dic[key][check_id]

        [
            dict_id[key][i].remove(j) 
            for i, j in zip(
                self.data_current.loc[inf, type], 
                inf
            ) if j in dict_id[key][int(i)]
        ]
        
        return True
    

    def current_infected(self, data_current):
        ''' создает словари id мест, где есть инфецированные '''

        hh_inf_dic = defaultdict(list)
        work_inf_dic = defaultdict(list)
        school_inf_dic = defaultdict(list)

        curr = data_current[data_current.infected > 0]
        print("All inf: ", len(curr))

        # TODO: сделать быстрее
        virulented = curr[curr.illness_day>2]

        for _, row in virulented.iterrows():
            ill_day = row.illness_day

            cur_key = self.StrainForIndex(row.infected)

            hh_inf_dic[row.sp_hh_id, cur_key].append(ill_day)
            if row.work_id != 0:
                if row.age > 17:
                    work_inf_dic[row.work_id, cur_key].append(ill_day)
                else:
                    school_inf_dic[row.work_id, cur_key].append(
                        ill_day) 

        return hh_inf_dic, work_inf_dic, school_inf_dic
    

    def make_dir(self, path, verbose):
        ''' создает дирикторию по пути, если нет '''

        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory created successfully!\n\n" if verbose else '', end='')
        else:
            print("Directory already exists!\n\n" if verbose else '', end='')

        return True


    def csv_from_dict(self, dic, f_name, number_seed): 
        ''' генерация csv файла из словаря '''
                    
        pd.DataFrame.from_dict(
            dic
        ).to_csv(
            self.results_dir + f_name.format(number_seed), 
            sep='\t',
            index=False
        )           

        return True 
    
    def csv_from_df(self, df, f_name, number_seed): 
        ''' генерация csv файла из датафрейма '''
                    
        df.to_csv(
            self.results_dir + f_name.format(number_seed), 
            sep='\t',
            index=False
        )           

        return True 

    def json_from_dict(self, dic, f_name, number_seed):
        ''' генерация json файла из словаря '''

        with open(self.results_dir + f_name.format(number_seed), "w") as outfile: 
            json.dump(dic, outfile)

    
    def IndexForStrain(self, strain):
        ''' получение индекса штамма по имени '''
        return self.strains_keys.index(strain) + 1
    

    def StrainForIndex(self, idx):
        ''' получение имени штамма по индексу '''

        return self.strains_keys[idx-1]
