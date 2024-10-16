from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm import tqdm
import random
from enum import Enum
import time


class Process(Enum):
    mortality = 1
    fertility = 2
    emigration = 3
    immigration = 4


@dataclass
class Population():
    """ Some variables for population data"""
    age_step = 5
    age_classes = 14
    data = pd.DataFrame({})
    age_max = 100
    age_year_step = [i for i in range(age_max+1)]
    age_classified = age_step*np.array([i for i in range(age_classes+1)])

    """ Annual characteristics """
    year_of_data = None
    annual_mortality = None
    annual_fertility = None
    annual_emigration = None
    annual_immigration = None

    # смертность мужчин на 1000 человек соответствующего возраста (по 5 лет)
    mortality_male = np.array([[1.5, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0],
                               [0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.1],
                               [0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3],
                               [0.5, 0.7, 0.6, 0.5, 0.7, 0.7, 0.8, 0.9],
                               [1.2, 1.2, 1.1, 1.1, 0.9, 1.1, 1.0, 1.2],
                               [2.7, 2.4, 2.1, 2.0, 1.9, 1.5, 1.4, 1.3],
                               [5.5, 4.9, 4.8, 4.2, 3.8, 3.1, 2.7, 2.6],
                               [6.3, 6.5, 6.2, 6.0, 5.9, 5.4, 4.9, 4.4],
                               [7.0, 6.7, 6.5, 6.5, 6.9, 7.0, 6.6, 6.5],
                               [9.5, 9.1, 7.9, 8.1, 7.9, 8.2, 7.9, 7.5],
                               [13.9, 12.7, 11.5, 11.1, 11.7, 11.5, 11.5, 10.3],
                               [20.3, 18.4, 16.9, 17.2, 16.7, 16.4, 15.6, 14.8],
                               [29.2, 27.6, 25.8, 25.6, 25.6, 25.5, 23.6, 23.6],
                               [36.8, 33.8, 33.3, 33.8, 34.5, 34.5, 33.7, 32.9],
                               [73.3, 75.3, 73.6, 72.1, 73.2, 71.7, 68.2, 67.0]]) / 1000
    # смертность женщин на 1000 человек соответствующего возраста (по 5 лет)
    mortality_female = np.array([[1.4, 1.2, 1.2, 1.2, 1.2, 1.0, 0.9, 0.7],  # смертность женщин на 1000 человек соответствующего возраста (по 5 лет)
                                 [0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1],
                                 [0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.2],
                                 [0.3, 0.3, 0.4, 0.4, 0.4, 0.3, 0.5, 0.4],
                                 [0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.4, 0.4],
                                 [1.0, 0.8, 0.7, 0.7, 0.8, 0.6, 0.5, 0.5],
                                 [1.4, 1.6, 1.5, 1.3, 1.4, 1.3, 1.1, 0.9],
                                 [2.0, 2.2, 2.0, 1.9, 2.1, 1.8, 1.8, 1.8],
                                 [2.7, 2.5, 2.4, 2.5, 2.6, 2.6, 2.3, 2.3],
                                 [3.5, 3.6, 3.2, 3.2, 3.2, 3.4, 3.0, 3.0],
                                 [5.0, 4.9, 4.4, 4.5, 4.5, 4.2, 4.1, 4.1],
                                 [7.6, 6.8, 6.7, 6.5, 6.6, 6.4, 6.0, 5.8],
                                 [11.0, 10.8, 10.6, 9.7, 10.2, 9.3, 9.2, 9.0],
                                 [14.8, 15.1, 14.6, 15.0, 15.0, 15.2, 14.6, 13.7],
                                 [59.0, 61.4, 61.3, 62.1, 63.9, 62.3, 60.7, 57.8]]) / 1000
    # рождаемость на 1000 женщин соотетствующего возраста (по 5 лет)
    fertility = np.array([0, 0, 0, 11.9, 53.0, 98.7, 73.3,
                         34.7, 7.3, 0.5, 0, 0, 0, 0, 0]) / 1000
    fertility = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [11.9, 12.3, 11.9, 11.8, 11.3, 9.7, 8.2, 7.6],
                          [53.0, 52.8, 50.4, 49.2, 51.1, 51.9, 48.2, 49.3],
                          [98.7, 104.4, 102.9, 102.4, 106.4, 104.9, 92.7, 87.0],
                          [73.3, 81.6, 83.5, 89.2, 94.4, 100.2, 91.7, 88.9],
                          [34.7, 40.2, 42.2, 45.5, 47.9, 51.9, 49.8, 50.0],
                          [7.3, 8.9, 9.0, 10.2, 11.3, 12.3, 12.3, 12.7],
                          [0.5, 0.4, 0.6, 0.7, 0.7, 1.0, 1.0, 1.1],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],]) / 1000
    # количество родившихся мальчиков/девочек
    birth_sex = np.array([[29.5, 32.3, 33.2, 34.6, 36.5, 37.3, 34.3, 32.9],
                          [27.5, 30.4, 31.0, 32.6, 34.2, 35.4, 32.2, 31.1]]) * 1000

    # смертность мужчин на 1000 человек соответствующего возраста (по 5 лет)
    emigration_male = np.array([[1.5, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0],
                               [0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.1],
                               [0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3],
                               [0.5, 0.7, 0.6, 0.5, 0.7, 0.7, 0.8, 0.9],
                               [1.2, 1.2, 1.1, 1.1, 0.9, 1.1, 1.0, 1.2],
                               [2.7, 2.4, 2.1, 2.0, 1.9, 1.5, 1.4, 1.3],
                               [5.5, 4.9, 4.8, 4.2, 3.8, 3.1, 2.7, 2.6],
                               [6.3, 6.5, 6.2, 6.0, 5.9, 5.4, 4.9, 4.4],
                               [7.0, 6.7, 6.5, 6.5, 6.9, 7.0, 6.6, 6.5],
                               [9.5, 9.1, 7.9, 8.1, 7.9, 8.2, 7.9, 7.5],
                               [13.9, 12.7, 11.5, 11.1, 11.7, 11.5, 11.5, 10.3],
                               [20.3, 18.4, 16.9, 17.2, 16.7, 16.4, 15.6, 14.8],
                               [29.2, 27.6, 25.8, 25.6, 25.6, 25.5, 23.6, 23.6],
                               [36.8, 33.8, 33.3, 33.8, 34.5, 34.5, 33.7, 32.9],
                               [73.3, 75.3, 73.6, 72.1, 73.2, 71.7, 68.2, 67.0]]) / 1000
    # смертность женщин на 1000 человек соответствующего возраста (по 5 лет)
    emigration_female = np.array([[1.4, 1.2, 1.2, 1.2, 1.2, 1.0, 0.9, 0.7],  # смертность женщин на 1000 человек соответствующего возраста (по 5 лет)
                                 [0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1],
                                 [0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.2],
                                 [0.3, 0.3, 0.4, 0.4, 0.4, 0.3, 0.5, 0.4],
                                 [0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.4, 0.4],
                                 [1.0, 0.8, 0.7, 0.7, 0.8, 0.6, 0.5, 0.5],
                                 [1.4, 1.6, 1.5, 1.3, 1.4, 1.3, 1.1, 0.9],
                                 [2.0, 2.2, 2.0, 1.9, 2.1, 1.8, 1.8, 1.8],
                                 [2.7, 2.5, 2.4, 2.5, 2.6, 2.6, 2.3, 2.3],
                                 [3.5, 3.6, 3.2, 3.2, 3.2, 3.4, 3.0, 3.0],
                                 [5.0, 4.9, 4.4, 4.5, 4.5, 4.2, 4.1, 4.1],
                                 [7.6, 6.8, 6.7, 6.5, 6.6, 6.4, 6.0, 5.8],
                                 [11.0, 10.8, 10.6, 9.7, 10.2, 9.3, 9.2, 9.0],
                                 [14.8, 15.1, 14.6, 15.0, 15.0, 15.2, 14.6, 13.7],
                                 [59.0, 61.4, 61.3, 62.1, 63.9, 62.3, 60.7, 57.8]]) / 1000

    # количество иммигрантов мужчин в каждой возрастной группе
    # immigration_male = np.array([4320,  4270,  2845, 14226, 23600, 22702, 17565, 12719, 10150,
    #                              8287,  6273,  4220,  2488,  1452,   717,   673,   482])
    immigration_male = np.ones((15, 8))*1000

    # количество иммигрантов женщин в каждой возрастной группе
    # immigration_female = np.array([4048,  4046,  2946, 12288, 17582, 20858, 15725, 10794,  8088,
    #                                6063,  5988,  4955,  3695,  2488,  1598,  1868,  1953])
    immigration_female = np.ones((15, 8))*1000

    def set_class_to_age(self, age):
        """ Set age clas according to age of person"""
        return (age//self.age_step) if age <= 70 else self.age_classes

    def read_data(self, year):
        """Read dataframe of population"""
        self.year_of_data = year
        self.data = pd.read_csv(
            r'people_{}.txt'.format(year), sep='\t', index_col=0)
        self.data = self.data[['sp_id', 'sp_hh_id', 'age', 'sex']]
        self.data[['sp_id', 'sp_hh_id', 'age']] = self.data[[
            'sp_id', 'sp_hh_id', 'age']].astype(int)
        self.data[['sex']] = self.data[['sex']].astype(str)
        self.data['age_class'] = list(
            map(lambda age: self.set_class_to_age(age), self.data.age))
        return

    def interpolate_population_coefficients(self, array_of_coefficients):
        coefficients_interpolated = [
            np.NaN for i in range(len(self.age_year_step))]
        y_data_index = 0

        for age in range(len(self.age_year_step)):
            if age % self.age_step == 0 and y_data_index < len(array_of_coefficients):
                coefficients_interpolated[age] = array_of_coefficients[y_data_index]
                y_data_index += 1
        coefficients_interpolated = pd.Series(
            coefficients_interpolated).interpolate()
        return coefficients_interpolated

    # used to remove people from population (mortality, emigration)
    def apply_remove_from_population(self, process, remove_coeff_male, remove_coeff_male_interpolated,
                                     remove_coeff_female, remove_coeff_female_interpolated):
        total_year_removed_data = 0
        for age_class in tqdm(range(self.age_classes+1)):
            male_remove_from = self.data.loc[(
                self.data['age_class'] == age_class) & (self.data['sex'] == 'M')]
            total_year_removed_data += len(male_remove_from) * \
                remove_coeff_male[age_class]
            female_remove_from = self.data.loc[(
                self.data['age_class'] == age_class) & (self.data['sex'] == 'F')]
            total_year_removed_data += len(female_remove_from) * \
                remove_coeff_female[age_class]
            total_year_removed_data = int(total_year_removed_data)

        total_year_removed_interpolated = 0
        remove_indexes_of_all_ages = []
        for age in range(self.age_max+1):
            current_age_people = self.data.loc[self.data['age'] == age]
            males_at_current_age = current_age_people.loc[current_age_people['sex'] == 'M']
            females_at_current_age = current_age_people.loc[current_age_people['sex'] == 'F']
            male_remove_from = len(males_at_current_age) * \
                remove_coeff_male_interpolated[age]
            female_remove_from = len(females_at_current_age) * \
                remove_coeff_female_interpolated[age]
            total_year_removed_interpolated += (
                male_remove_from + female_remove_from)
            remove_male_indexes = np.random.choice(np.array(males_at_current_age.sp_id),
                                                   int(round(
                                                       male_remove_from)),
                                                   replace=False)
            remove_female_indexes = np.random.choice(np.array(females_at_current_age.sp_id),
                                                     int(round(
                                                         female_remove_from)),
                                                     replace=False)
            remove_indexes_of_all_ages.extend(remove_male_indexes)
            remove_indexes_of_all_ages.extend(remove_female_indexes)
        total_year_removed_interpolated = int(
            total_year_removed_interpolated)

        # print("Total year mortality calculated by 5 year groups coefficients:",
        #       total_year_removed_data)
        # print("Total year mortality calculated by interpolated coefficients:",
        #       total_year_removed_interpolated)

        # remove deceased or emigrated people from population
        self.data = self.data[~self.data.sp_id.isin(
            remove_indexes_of_all_ages)]
        # print("Population after mortality:", self.data.shape[0])

        if process == Process['mortality']:
            self.annual_mortality = total_year_removed_interpolated
        if process == Process['emigration']:
            self.annual_emigration = total_year_removed_interpolated

    def apply_fertility(self, fertility_interpolated, birth_sex):
        probability_male = birth_sex[0]/sum(birth_sex)
        probabilty_female = 1 - probability_male

        fertility_indexes = []
        for age in range(self.age_max+1):
            people_at_current_age = self.data.loc[self.data.age == age]
            females_at_current_age = people_at_current_age.loc[people_at_current_age['sex'] == 'F']
            fertility_at_current_age = np.random.choice(np.array(females_at_current_age.sp_id), int(
                round(fertility_interpolated[age]*len(females_at_current_age))), replace=False)
            fertility_indexes.extend(fertility_at_current_age)

        total_fertility_male = int(len(fertility_indexes)*probability_male)
        total_fertility_female = int(len(fertility_indexes)*probabilty_female)

        random.shuffle(fertility_indexes)
        fertility_male_indexes = fertility_indexes[:total_fertility_male]
        fertility_female_indexes = fertility_indexes[total_fertility_male:]

        id_list_male = list(range(max(self.data.sp_id) + 1,
                                  max(self.data.sp_id) + total_fertility_male + 1))
        age_list_male = [0] * total_fertility_male
        sex_list_male = ['M'] * total_fertility_male

        hh_list_male = self.data.loc[self.data['sp_id'].isin(
            fertility_male_indexes[:total_fertility_male])]
        hh_list_male = hh_list_male['sp_hh_id'].to_numpy()

        new_frame_male = pd.DataFrame({'sp_id': id_list_male, 'sp_hh_id': hh_list_male,
                                       'age': age_list_male, 'sex': sex_list_male, 'age_class': age_list_male})
        self.data = pd.concat([self.data, new_frame_male], ignore_index=True)

        id_list_female = list(range(max(self.data.sp_id) + 1,
                                    max(self.data.sp_id) + total_fertility_female + 1))
        age_list_female = [0] * total_fertility_female
        sex_list_female = ['F'] * total_fertility_female

        hh_list_female = self.data.loc[self.data['sp_id'].isin(
            fertility_female_indexes[:total_fertility_female])]
        hh_list_female = hh_list_female['sp_hh_id'].to_numpy()

        new_frame_female = pd.DataFrame({'sp_id': id_list_female, 'sp_hh_id': hh_list_female,
                                         'age': age_list_female, 'sex': sex_list_female, 'age_class': age_list_female})
        self.data = pd.concat([self.data, new_frame_female], ignore_index=True)

        # annual number
        self.annual_fertility = len(hh_list_female) + len(hh_list_male)
        return

    def apply_immigration(self, immigration_male, immigration_male_interpolated,
                          immigration_female, immigration_female_interpolated):
        immigration_male_interpolated = [
            int(i/100) for i in immigration_male_interpolated]
        immigration_female_interpolated = [
            int(i/100) for i in immigration_female_interpolated]
        hh_id_array = self.data['sp_hh_id'].unique()
        total_year_immigration = 0
        immigrants_dataframe = pd.DataFrame(
            columns=['sp_id', 'sp_hh_id', 'age', 'sex', 'age_class'])
        for age in tqdm(range(self.age_max+1)):
            # males
            id_males_for_immigrants = list(range(int(max(self.data.sp_id) + 1),
                                                 int(max(self.data.sp_id) + immigration_male_interpolated[age] + 1)))
            hh_id_list_for_male_at_current_age = np.random.choice(
                np.array(hh_id_array), int(immigration_male_interpolated[age]), replace=False).tolist()
            age_list_male = [age] * int(immigration_male_interpolated[age])
            sex_list_male = ['M'] * int(immigration_male_interpolated[age])

            new_frame_male = pd.DataFrame({'sp_id': id_males_for_immigrants, 'sp_hh_id': hh_id_list_for_male_at_current_age,
                                           'age': age_list_male, 'sex': sex_list_male, 'age_class': age_list_male})
            immigrants_dataframe = pd.concat(
                [immigrants_dataframe, new_frame_male], ignore_index=True)

            # females
            id_females_for_immigrants = list(range(int(max(self.data.sp_id) + 1),
                                                   int(max(self.data.sp_id) + immigration_female_interpolated[age] + 1)))
            hh_id_list_for_female_at_current_age = np.random.choice(
                np.array(hh_id_array), int(immigration_female_interpolated[age]), replace=False).tolist()
            age_list_female = [age] * int(immigration_female_interpolated[age])
            sex_list_female = ['F'] * int(immigration_female_interpolated[age])

            new_frame_female = pd.DataFrame({'sp_id': id_females_for_immigrants, 'sp_hh_id': hh_id_list_for_female_at_current_age,
                                             'age': age_list_female, 'sex': sex_list_female, 'age_class': age_list_female})
            immigrants_dataframe = pd.concat(
                [immigrants_dataframe, new_frame_female], ignore_index=True)

            total_year_immigration += (
                immigration_male_interpolated[age] + immigration_female_interpolated[age])

        self.data = pd.concat(
            [self.data, immigrants_dataframe], ignore_index=True)
        self.annual_immigration = total_year_immigration

    def display_data(self):
        display(self.data)

    def display_data_info(self):
        print("----------Population data info----------")
        print("Year of data: {}".format(self.year_of_data))
        print("Population at the end of the year: {}".format(
            self.data.shape[0]))
        print("Annual mortality: {}".format(self.annual_mortality))
        print("Annual fertility: {}".format(self.annual_fertility))
        print("Annual emigration: {}".format(self.annual_emigration))
        print("Annual immigration: {}".format(self.annual_immigration))
        print("----------------------------------------")


def generate_future_population(Population, year=2010):
    year_index = year - 2010  # because of data starts only from 2010 year
    mortality_male_interpolated = Population.interpolate_population_coefficients(
        Population.mortality_male[:, year_index])
    mortality_female_interpolated = Population.interpolate_population_coefficients(
        Population.mortality_female[:, year_index])
    Population.apply_remove_from_population(Process['mortality'], Population.mortality_male[:, year_index], mortality_male_interpolated,
                                            Population.mortality_female[:, year_index], mortality_female_interpolated)
    fertility_interpolated = Population.interpolate_population_coefficients(
        Population.fertility[:, year_index])
    Population.apply_fertility(
        fertility_interpolated, Population.birth_sex[:, year_index])
    emigration_male_interpolated = Population.interpolate_population_coefficients(
        Population.emigration_male[:, year_index])
    emigration_female_interpolated = Population.interpolate_population_coefficients(
        Population.emigration_female[:, year_index])
    Population.apply_remove_from_population(Process['emigration'], Population.emigration_male[:, year_index],
                                            emigration_male_interpolated, Population.emigration_female[
                                                :, year_index],
                                            emigration_female_interpolated)
    immigration_male_interpolated = Population.interpolate_population_coefficients(
        Population.immigration_male[:, year_index])
    immigration_female_interpolated = Population.interpolate_population_coefficients(
        Population.immigration_female[:, year_index])
    Population.apply_immigration(Population.immigration_male[:, year_index], immigration_male_interpolated,
                                 Population.immigration_female[:, year_index], immigration_female_interpolated)
    return


if __name__ == '__main__':
    start_time = time.time()
    population_data_by_years = pd.DataFrame(
        columns=['year', 'population', 'mortality', 'fertility', 'emigration', 'immigration'])

    for year in tqdm(range(2010, 2018)):
        Population = Population()
        Population.read_data(year=year)
        generate_future_population(Population, year)
        list_of_annual_data = [year, Population.data.shape[0], Population.annual_mortality,
                               Population.annual_fertility, Population.annual_emigration, Population.annual_immigration]
        population_data_by_years.loc[len(
            population_data_by_years)] = list_of_annual_data
    population_data_by_years.to_csv('annual_data.csv')
    print("Time of execution: {} min".format(
        float(time.time() - start_time)/60))


# Population.display_data_info()
# print("Time of execution: {} min".format(float(time.time() - start_time)/60))
