import pandas as pd
import numpy as np
from scipy import stats
import warnings
import datetime
import random
import json
import os
import copy
import time
import matplotlib.pyplot as plt
from statistics import mean
from functools import partial
from collections import defaultdict
import multiprocessing as mp
warnings.filterwarnings('ignore')
pd.options.display.width = None
pd.options.display.max_columns = None

# forlorn_v2 - вторичное использование случайных чисел из массива данных
# multistrain - несколько штаммов
# _v2 - refactored

data = pd.read_csv(r'synth_populations/people_apart+work.txt',
                   sep='\t', index_col=0)
data = data[['sp_id', 'sp_hh_id', 'age', 'sex', 'work_id']]
households = pd.read_csv(r'synth_populations/households_apart.txt', sep='\t')
households = households[['sp_id', 'latitude', 'longitude']]
dict_school_id_global = json.load(
    open(os.path.expanduser(r'synth_populations/school_id_5-15km.json')))
dict_school_len = [len(dict_school_id_global[i])
                   for i in dict_school_id_global.keys()]


# Изначальное количество больных в городе
infected_init_dic = {'H1N1': 10, 'H3N2': 10, 'B': 10}
# 0.05 0.275 #0.5 0.725 #0.95
alpha_dic = {'H1N1': 0.57, 'H3N2': 0.13, 'B': 0.1}
# Воронеж  {'H1N1': 0.57, 'H3N2': 0.13, 'B': 0.1} #Москва {'H1N1': 0.78, 'H3N2': 0.74, 'B': 0.6}

lmbd = 0.3
num_runs = 1
days = range(1, 50)

strains_keys = ['H1N1', 'H3N2', 'B']


def strainForInfIndex(idx):  # idx from 1
    return strains_keys[idx-1]


def infIndexForStrain(strain):
    return strains_keys.index(strain) + 1


def aggregateOutputDics(dic_list, strains_keys):
    dic_res = {}
    for key in strains_keys:
        dic_res[key] = []
        for elem in dic_list:
            dic_res[key].append(elem[key])

    return dic_res


def func_b_r(inf_day):
    a = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]
    if inf_day < 9:
        return a[inf_day - 1]
    else:
        return 0


def main_function(number_seed, dataset, dict_school_id_all):
    np.random.seed(number_seed)

    data_current = dataset.copy()
    y = {}

    print("Main function")
    start_time = time.time()

    # Штаммы отличаются цифрой в поле infected (от 1 до 3)
    for key in strains_keys:
        # data_susceptible_strain = dataset[dataset["susceptible_" + key] == 1] #Все восприимчивые к данному штамму
        y = np.random.choice(np.array(dataset[dataset["susceptible_"+key] == 1].sp_id),
                             infected_init_dic[key], replace=False)  # Выбираем отсюда же ID больных

        data_current.loc[pd.np.in1d(data_current.sp_id, y), ['infected', 'susceptible_'+str(
            key), 'illness_day']] = [infIndexForStrain(key), 0, 3]  # Раскидываем больных

    # print(data_susceptible[data_susceptible.infected > 0].drop_duplicates())

    id_susceptible_list_dic, latitude_list_dic, longitude_list_dic, type_list_dic, id_place_list_dic, days_inf_dic, results_dic = {}, {}, {}, {}, {}, {}, {}

    for key in strains_keys:
        id_susceptible_list_dic[key], latitude_list_dic[key], longitude_list_dic[key], type_list_dic[
            key], id_place_list_dic[key], days_inf_dic[key], results_dic[key] = [], [], [], [], [], [], []

    # dict_school_id_all.copy() #Первоначальный список домохозяйств с инфицированными
    dict_hh_id = copy.deepcopy(dict_school_id_all)
    dict_hh_id.clear()

    for key in strains_keys:
        dict_hh_id[key] = {i: list(data_current[(data_current.sp_hh_id == i) & (data_current["susceptible_"+key] == 1)].index)  # Восприимчивые в местах, где есть инфицированные
                           for i in data_current.loc[data_current.infected == infIndexForStrain(key), 'sp_hh_id']}  # конкретным штаммом

    # dict_school_id_all.copy()
    dict_work_id = copy.deepcopy(dict_school_id_all)
    dict_work_id.clear()

    for key in strains_keys:
        dict_work_id[key] = {int(i): list(data_current[(data_current.age > 17) & (data_current.work_id == i) & (data_current["susceptible_"+key] == 1)].index)
                             for i in data_current.loc[(data_current.infected == infIndexForStrain(key)) & (data_current.age > 17) & (data_current.work_id != 'X'), 'work_id']}

    [dict_school_id_all[str(i)].remove(j) for i, j in zip(data_current.loc[(data_current.infected > 0) & (data_current.age <= 17) &
                                                                           (data_current.work_id != 'X'), 'work_id'],
                                                          data_current[(data_current.infected > 0) & (data_current.age <= 17) &
                                                                       (data_current.work_id != 'X')].index)]

    # Сначала были все дети в школах, сейчас убрали больных и остались потенциально заразные и иммунные. Нужно убрать иммунных к данному штамму для каждого подсловаря.

    dict_school_id = {}  # Здесь будут списки восприимчивых в местах по штаммам

    # Убираем из школы i детишек j тех, что иммунны к key (оставляем восприимчивых)
    for key in strains_keys:
        dict_school_id[key] = copy.deepcopy(dict_school_id_all)
        [dict_school_id[key][str(i)].remove(j) for i, j in
         zip(data_current.loc[(data_current["susceptible_"+key] == 0) & (data_current.infected == 0) & (data_current.age <= 17) &
                              (data_current.work_id != 'X'), 'work_id'],
             data_current[(data_current["susceptible_"+key] == 0) & (data_current.infected == 0) & (data_current.age <= 17) &
                          (data_current.work_id != 'X')].index)]

    vfunc_b_r = np.vectorize(func_b_r)

    # print("Starting from I0: ")
    # print(data_current[data_current.illness_day > 2].sp_id)

    time0 = time.time()
    print("Init, time elapsed: {}".format(time0-start_time))

    for j in days:  # Главный цикл
        time1 = time.time()
        print("Day ", j)
        # Есть заразные !!MAGIC NUMBER!! #TODO: убрать
        if len(data_current[data_current.illness_day > 2]) != 0:
            # TODO: генерация с.в. при расходе
            x_rand = np.random.rand(10000000)

            # Заражены каким-то штаммом, нужен для цикла сбора протяженности инфекции
            curr = data_current[data_current.infected > 0]
            print("All inf: ", len(curr))

            hh_inf_dic, work_inf_dic, school_inf_dic = defaultdict(list), defaultdict(list), defaultdict(
                list)

            for _, row in curr.iterrows():  # Сбор id всяких мест с инфицированными
                ill_day = row.illness_day
                if ill_day > 2:
                    # strains_keys[row.infected-1] #0 - H1, 1 - H3, 2 - B   #смотрим, чем болен
                    cur_key = strainForInfIndex(row.infected)
                    # Сбор id домохозяйств с инфицированными
                    hh_inf_dic[row.sp_hh_id, cur_key].append(ill_day)
                    if row.work_id != 'X':
                        if row.age > 17:
                            work_inf_dic[row.work_id, cur_key].append(ill_day)
                        else:
                            school_inf_dic[row.work_id, cur_key].append(
                                ill_day)  # Сбор id школ с инфицированными

            time2 = time.time()

            real_inf_hh_dic = {}  # Здесь будем регистрировать вновь заболевших с разбивкой по штаммам
            for key in strains_keys:
                real_inf_hh_dic[key] = np.array([])

            hh_inf_dic_keys_shuffled = list(hh_inf_dic.keys())
            random.shuffle(hh_inf_dic_keys_shuffled)

            # Все домохозяйства, где есть инфицированные любым из штаммов (hh_id, strain)
            for hh_infection_key in hh_inf_dic_keys_shuffled:

                i = hh_infection_key[0]  # id домохозяйства
                cur_strain = hh_infection_key[1]  # штамм

                # Нет данных о жителях в словаре по id домохозяйства для данного штамма (бо добавились инфекции)
                if i not in dict_hh_id[cur_strain].keys():
                    dict_hh_id[cur_strain].update(
                        {i: list(data_current[(data_current.sp_hh_id == i)].index)})
                # Все восприимчивые к конкретному штамму люди в данном домохозяйстве
                hh_len = len(dict_hh_id[cur_strain][i])
                if hh_len != 0:

                    # Инфекционность для всех больных c данным штаммом
                    temp = vfunc_b_r(hh_inf_dic[hh_infection_key])

                    # Самая тяжелая процедура здесь
                    prob = np.repeat(temp, hh_len) * lmbd
                    curr_length = len(prob)
                    hh_rand = x_rand[:curr_length]
                    # Убираем в хвост массива потраченные случайные числа
                    np.roll(x_rand, curr_length)
                    real_inf = len(hh_rand[hh_rand < prob])
                    if hh_len < real_inf:
                        real_inf = hh_len

                    ##################

                    # Выбираем, кто заразился, из восприимчивых
                    real_inf_id = np.random.choice(
                        np.array(dict_hh_id[cur_strain][i]), real_inf, replace=False)
                    # Дописываем инфицированных по штаммам и домохозяйствам
                    real_inf_hh_dic[cur_strain] = np.concatenate(
                        (real_inf_hh_dic[cur_strain], real_inf_id))

                    id_susceptible_list_dic[cur_strain].extend(
                        data_current.sp_id[real_inf_id])  # Выходной txt, списки заболевших
                    type_list_dic[cur_strain].extend(
                        ['household'] * len(real_inf_id))
                    id_place_list_dic[cur_strain].extend(
                        data_current.sp_hh_id[real_inf_id])  # Места, где заболели
                    days_inf_dic[cur_strain].extend([j] * len(real_inf_id))

            time3 = time.time()
            print("Households, time elapsed: {}".format(time3 - time2))

            real_inf_work_dic = {}

            for key in strains_keys:
                real_inf_work_dic[key] = np.array([])

            some_current = data_current[(data_current.work_id != 'X') & (
                data_current.age > 17)]  # Подможество работающих из всей популяции
            some_current[['work_id']] = some_current[['work_id']].astype(int)

            work_inf_dic_keys_shuffled = list(work_inf_dic.keys())
            random.shuffle(work_inf_dic_keys_shuffled)

            for work_infection_key in work_inf_dic_keys_shuffled:
                i = work_infection_key[0]  # id работы
                cur_strain = hh_infection_key[1]  # штамм

                if i not in dict_work_id[cur_strain].keys():  # Зачем?
                    dict_work_id[cur_strain].update(
                        {i: list(some_current[some_current.work_id == int(i)].index)})

                # Кол-во восприимчивых в месте
                work_len = len(dict_work_id[cur_strain][i])

                if work_len != 0:
                    # Назначаем инфекционность   #Берём инфицированных
                    temp = vfunc_b_r(work_inf_dic[work_infection_key])

                    # Вер. встретить и заразить (lambda*g(tau))
                    prob = np.repeat(temp, work_len) * lmbd
                    curr_length = len(prob)
                    work_rand = x_rand[:curr_length]
                    # Убираем в хвост массива потраченные случайные числа
                    np.roll(x_rand, curr_length)

                    real_inf = len(work_rand[work_rand < prob])
                    if work_len < real_inf:
                        real_inf = work_len
                    real_inf_id = np.random.choice(
                        np.array(dict_work_id[cur_strain][i]), real_inf, replace=False)
                    real_inf_work_dic[cur_strain] = np.concatenate(
                        (real_inf_work_dic[cur_strain], real_inf_id))

                    id_susceptible_list_dic[cur_strain].extend(
                        data_current.sp_id[real_inf_id])
                    type_list_dic[cur_strain].extend(
                        ['workplace'] * len(real_inf_id))

                    id_place_list_dic[cur_strain].extend(
                        map(lambda x: int(x), data_current.work_id[real_inf_id]))
                    days_inf_dic[cur_strain].extend([j] * len(real_inf_id))

            time4 = time.time()
            print("Workplaces, time elapsed: {}".format(time4 - time3))

            real_inf_school_dic = {}  # Инфекционные в школе и сами такие школы

            for key in strains_keys:
                real_inf_school_dic[key] = np.array([])

            school_inf_dic_keys_shuffled = list(school_inf_dic.keys())
            random.shuffle(school_inf_dic_keys_shuffled)

            for school_infection_key in school_inf_dic_keys_shuffled:
                i = school_infection_key[0]  # id школы
                cur_strain = hh_infection_key[1]  # штамм
                school_len = len(dict_school_id[cur_strain][str(i)])
                if school_len != 0:
                    length = dict_school_len[list(dict_school_id[cur_strain].keys()).index(
                        str(i))]  # Общее число людей
                    temp = vfunc_b_r(school_inf_dic[school_infection_key])
                    # Вероятность контакта фикс индивида с другим
                    prob_cont = 8.5 / (length - 1) if (8.5 + 1) < length else 1
                    res = np.prod(1 - prob_cont * lmbd * temp)
                    real_inf = np.random.binomial(length - 1, 1 - res)
                    if school_len < real_inf:
                        real_inf = school_len

                    real_inf_id = np.random.choice(
                        np.array(dict_school_id[cur_strain][str(i)]), real_inf, replace=False)
                    real_inf_school_dic[cur_strain] = np.concatenate(
                        (real_inf_school_dic[cur_strain], real_inf_id))

                    id_susceptible_list_dic[cur_strain].extend(
                        data_current.sp_id[real_inf_id])
                    type_list_dic[cur_strain].extend(
                        ['school'] * len(real_inf_id))

                    id_place_list_dic[cur_strain].extend(
                        map(lambda x: int(x), data_current.work_id[real_inf_id]))  # work_id
                    days_inf_dic[cur_strain].extend([j] * len(real_inf_id))
            # print(number_seed, 'schools', len(school_inf), datetime.datetime.now())

            real_inf_results_dic = {}

            for key in strains_keys:  # Собираем в одном месте всех, кто заболел
                real_inf_results_dic[key] = np.concatenate(
                    (real_inf_hh_dic[key], real_inf_school_dic[key], real_inf_work_dic[key]))
                real_inf_results_dic[key] = np.unique(
                    real_inf_results_dic[key].astype(int))

            # Считаем, что при множественном заражении одного человека "побеждает" один из вирусов
            # Remove??
            # real_inf_results_intersection = list(set(real_inf_results_dic[strains_keys[0]]) & set(real_inf_results_dic[strains_keys[1]]) & set(real_inf_results_dic[strains_keys[2]]))
            # print("Intersections: ".format(real_inf_results_intersection))
            # for id in real_inf_results_intersection:
            #     infByStrain = [i for i in range(1,4) if id in real_inf_results_dic[strainForInfIndex(i)]]
            #     print("Intersection: {}".format(infByStrain))
            #     ultimateStrainToBlame = random.choice(infByStrain)
            #     infByStrain.remove(ultimateStrainToBlame)
            #     print("Removing: {}".format(infByStrain))
            #     for i in infByStrain:
            #         real_inf_results_dic[strainForInfIndex(i)].remove(id)

            ################

            time5 = time.time()
            print("Schools, time elapsed: {}".format(time5 - time4))

            for key in strains_keys:
                # strains_keys.index[cur_strain] + 1 #порядковый номер штамма для графы infected
                data_current.loc[real_inf_results_dic[key], [
                    'infected', 'illness_day', 'susceptible_'+key]] = [infIndexForStrain(key), 1, 0]

            for key in strains_keys:  # Обновление словарей мест с id
                current_hh_id = []
                [current_hh_id.extend(i) for i in dict_hh_id[key].values()]
                check_id = [
                    True if i in current_hh_id else False for i in real_inf_results_dic[key]]
                check_id = [i for i, x in enumerate(check_id) if x is True]
                # Инфицированные домохозяйства
                inf_hh = real_inf_results_dic[key][check_id]
                [dict_hh_id[key][i].remove(j) for i, j in zip(
                    data_current.loc[inf_hh, 'sp_hh_id'], inf_hh)]

                current_wp_id = []
                [current_wp_id.extend(i) for i in dict_work_id[key].values()]
                check_id = [
                    True if i in current_wp_id else False for i in real_inf_results_dic[key]]
                check_id = [i for i, x in enumerate(check_id) if x is True]
                inf_wp = real_inf_results_dic[key][check_id]
                # TODO: if j in dict_work_id[key][i] is necessary?
                [dict_work_id[key][i].remove(j) for i, j in zip(
                    data_current.loc[inf_wp, 'work_id'], inf_wp) if j in dict_work_id[key][i]]

                inf_school = real_inf_results_dic[key][(data_current.loc[real_inf_results_dic[key], 'work_id'] != 'X') & (
                    data_current.loc[real_inf_results_dic[key], 'age'] <= 17)]
                [dict_school_id[key][str(i)].remove(j) for i, j in zip(data_current.loc[inf_school, 'work_id'], inf_school)
                 if j in dict_school_id[key][i]]  # TODO: if j in dict_school_id[key][i] is necessary?

            time6 = time.time()
            print("Day results, time elapsed: {}".format(time6 - time5))

        for key in strains_keys:
            newly_infected_by_strain = len(data_current[(data_current.illness_day == 1) & (
                data_current.infected == infIndexForStrain(key))])
            results_dic[key].append(newly_infected_by_strain)

            dataset_infected_by_strain = data_current[data_current.infected == infIndexForStrain(key)][[
                'sp_id', 'sp_hh_id']]
            dataset_infected_by_strain = dataset_infected_by_strain.sort_values(by=[
                                                                                'sp_hh_id'])
            temp = households.loc[households.index.intersection(
                dataset_infected_by_strain.sp_hh_id), ['latitude', 'longitude']]

            temp.index = dataset_infected_by_strain.index
            dataset_infected_by_strain['latitude'] = temp.latitude
            dataset_infected_by_strain['longitude'] = temp.longitude
            dataset_infected_by_strain[['sp_id', 'latitude', 'longitude']].to_csv(
                r'results/infected_{}_seed_{}_day_{}.txt'.format(number_seed, j, key), sep='\t', index=False)

            print("{}: day {} {} I {}({}) S {} Time {}".format(number_seed, j, key, newly_infected_by_strain, int(data_current[data_current.infected == infIndexForStrain(key)]['infected'].sum()),
                  int(data_current[['susceptible_'+key]].sum()), datetime.datetime.now()))
        print()

        data_current.loc[data_current.infected >
                         0, 'illness_day'] += 1  # Time goes
        data_current.loc[data_current.illness_day > 8, [
            'infected', 'illness_day']] = 0  # Recovery

        # endif j in days
    dataset_place_dic = {}

    # if not os.path.exists("results/"):
    os.makedirs("results/", exist_ok=True)

    for key in strains_keys:
        dataset_place_dic[key] = pd.DataFrame({'day': days_inf_dic[key], 'strain': key, 'sp_id': id_susceptible_list_dic[key],
                                              'place_type':  type_list_dic[key], 'place_id': id_place_list_dic[key]})

        dataset_place_dic[key].to_csv(r'results/{}_newly_infected_place_{}.txt'.format(number_seed, key),
                                      sep='\t', index=False)
    # [strains_keys[0]], results_dic[strains_keys[1]], results_dic[strains_keys[2]]
    return results_dic


if __name__ == '__main__':
    data[['sp_id', 'sp_hh_id', 'age']] = data[[
        'sp_id', 'sp_hh_id', 'age']].astype(int)
    data[['work_id']] = data[['work_id']].astype(str)
    data = data.sample(frac=1)  # frac=1
    households[['sp_id']] = households[['sp_id']].astype(int)
    households[['latitude', 'longitude']] = households[[
        'latitude', 'longitude']].astype(float)
    households.index = households.sp_id

    for key in strains_keys:
        data['susceptible_'+key] = 0

    data['infected'] = 0
    data['illness_day'] = 0

    cpu_num = mp.cpu_count()

    print("{} processors detected".format(cpu_num))

    for key in strains_keys:
        data.loc[np.random.choice(data.index, round(
            len(data) * alpha_dic[key]), replace=False), 'susceptible_'+key] = 1

    # Serial for debugging
    output = main_function(
        dataset=data, dict_school_id_all=dict_school_id_global, number_seed=1)

    # output = {}
    #
    # with mp.Pool(cpu_num-2) as pool:
    #     output = pool.map(partial(main_function, dataset=data, dict_school_id_all = dict_school_id_global), range(num_runs))
    # result = aggregateOutputDics(output, strains_keys)
    #
    # for key in strains_keys:
    #     output_cur = result[key]
    #     print(output_cur)
    #     mean_ = [*map(mean, zip(*output_cur))]
    #     print(mean_)
    #     print([*map(min, zip(*output_cur))])
    #     print([*map(max, zip(*output_cur))])
    #
    #     plt.plot(days, mean_, color='red')
    #     plt.plot(days, [*map(min, zip(*output_cur))], color='green')
    #     plt.plot(days, [*map(max, zip(*output_cur))], color='green')
    #     plt.legend(('Mean', 'Min', 'Max'), loc='upper right')
    #     plt.xlabel('Duration of days')
    #     plt.ylabel('Active incidence cases')
    #     plt.show()
