import numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, pool) -> None:
        ''' класс для предобработки основных данных '''

        self.pool = pool

        self.load_data()
        self.preprocess_data()
        self.set_initial_values()
        self.vaccination()
        self.init_infected()
        self.preprocess_places()
    

    def load_data(self):
        ''' загрузка данных '''
        # np.random.seed(1)

        pool = self.pool

        data = pd.read_csv(
            pool.data_path + 'people.txt', 
            sep='\t', index_col=0
        )

        data = data[['sp_id', 'sp_hh_id', 'age', 'sex', 'work_id']]

        pool.dict_school_len = {int(i[0]): len(i[1]) for i in data[(data.age<18)&(data.work_id!='X')].groupby('work_id')}

        pool.data_current = data

        self.pool = pool

        return True
    

    def preprocess_data(self):
        ''' предобработка данных '''

        # np.random.seed(1)

        pool = self.pool

        data = pool.data_current


        data[['sp_id', 'sp_hh_id', 'age']] = data[
            ['sp_id', 'sp_hh_id', 'age']
        ].astype(int)

        data[['sex']] = data[['sex']].replace(['F', 'M'], ['0', '1'])
        data[['sex']] = data[['sex']].astype(int)

        data[['work_id']] = data[['work_id']].replace(['X'], ['0'])
        data[['work_id']] = data[['work_id']].astype(int)
        data = data.sample(frac=1)

        pool.data_current = data
        self.pool = pool
        return True
    

    def set_initial_values(self):
        ''' задание первоначально зараженных '''

        # np.random.seed(1)
        
        pool = self.pool

        data = pool.data_current

        for key in pool.strains_keys:
            data['susceptible_'+key] = 0

        data['infected'] = 0
        data['illness_day'] = 0

        for key in pool.strains_keys:
            data.loc[
                np.random.choice(
                    data.index, 
                    round(len(data) * pool.alpha_dic[key]),
                    replace=False
                ),         
                'susceptible_'+key
            ] = 1

        pool.data_current = data
        self.pool = pool
        return True
    

    def vaccination(self):
        ''' вакцинация населения '''

        # np.random.seed(1)

        pool = self.pool

        data = pool.data_current
        age_groups = pool.age_groups
        vaccined_fraction = pool.vaccined_fraction

        if len(age_groups)!=len(vaccined_fraction):
            raise Exception(
                "Размерность списка списка возрасных групп и " +
                "списка долей вакцинирования этих групп не совпадает"
            )
            
        y = np.array([])
        
        for i in range(len(vaccined_fraction)):
            ages = age_groups[i].split('-')

            group = data[
                (data.age >= int(ages[0])) &
                (data.age <= int(ages[1]))
            ]          

            y = np.concatenate((
                    np.random.choice(
                        group.index, 
                        round(len(group)*vaccined_fraction[i]/100),
                        replace=False
                    ),
                    y
            ))

        for i in pool.strains_keys:
            data.loc[data.index.isin(y), f'susceptible_{i}'] = 0

        pool.data_current = data
        self.pool = pool
        return True
    

    def init_infected(self):
        ''' задание первоначально больных '''

        # np.random.seed(1)

        pool = self.pool
        data = pool.data_current

        for key in pool.strains_keys:
            y = np.random.choice(
                np.array(data[data["susceptible_"+key] == 1].sp_id),
                pool.infected_init_dic[key], 
                replace=False
            )  

            data.loc[
                np.in1d(
                    data.sp_id,
                    y
                ), 
                ['infected', 'susceptible_'+str(key), 'illness_day']
            ] = [pool.IndexForStrain(key), 0, 3]  
        
        pool.data_current = data
        self.pool = pool
        return True
    

    def preprocess_places(self):
        ''' создание словарей с восприимчивыми в местах, где есть инфицированные конкретным штаммом '''

        pool = self.pool
        data_current = pool.data_current
                           
        pool.dict_hh_id = dict()
        pool.dict_work_id = dict()
        pool.dict_school_id = dict()                

        for key in pool.strains_keys:
            IndexForStrain = pool.IndexForStrain
            
            pool.dict_hh_id[key] = {i: list(
                                            data_current[
                                                (data_current.sp_hh_id == i) &
                                                (data_current["susceptible_"+key] == 1)
                                            ].index
                                       ) for i in data_current.loc[
                                        data_current.infected == IndexForStrain(key), 
                                        'sp_hh_id'
                                    ]}

            pool.dict_work_id[key] = {i: list(
                                            data_current[
                                                (data_current.age > 17) &
                                                (data_current.work_id == i) &
                                                (data_current["susceptible_"+key] == 1)
                                            ].index
                                        ) for i in data_current.loc[
                                    (data_current.infected == IndexForStrain(key)) &
                                        (data_current.age > 17) &
                                    (data_current.work_id != 0), 
                                    'work_id'
                            ]}

            pool.dict_school_id[key] = {i: list(
                                                data_current[
                                                    (data_current.age < 18) & 
                                                    (data_current.work_id == i) & 
                                                    (data_current["susceptible_"+key] == 1)
                                                ].index
                                            ) for i in data_current.loc[
                                    (data_current.infected == IndexForStrain(key)) & 
                                        (data_current.age < 18) & 
                                    (data_current.work_id != 0), 
                                    'work_id'
                            ]}
        return True
