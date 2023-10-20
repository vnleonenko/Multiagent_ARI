from typing import List
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import BSpline, make_interp_spline
from IPython.display import display
from tqdm import tqdm
import time
from FPOP_param import *
import numpy as np
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
# import datetime
import warnings
warnings.filterwarnings('ignore')


class FPOP:

    def __init__(self) -> None:
        self.age_step = 5

        self.mortality_male_interpolated = []
        self.mortality_female_interpolated = []

        self.mortality = [61732, 62025, 60308, 60218, 61996, 61552,
                          60690, 59844]

        self.mortality_male = np.array([[1.5, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0],  # смертность мужчин на 1000 человек соответствующего возраста (по 5 лет)
                                        [0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.1],
                                        [0.3, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3],
                                        [0.5, 0.7, 0.6, 0.5, 0.7, 0.7, 0.8, 0.9],
                                        [1.2, 1.2, 1.1, 1.1, 0.9, 1.1, 1.0, 1.2],
                                        [2.7, 2.4, 2.1, 2.0, 1.9, 1.5, 1.4, 1.3],
                                        [5.5, 4.9, 4.8, 4.2, 3.8, 3.1, 2.7, 2.6],
                                        [6.3, 6.5, 6.2, 6.0, 5.9, 5.4, 4.9, 4.4],
                                        [7.0, 6.7, 6.5, 6.5, 6.9, 7.0, 6.6, 6.5],
                                        [9.5, 9.1, 7.9, 8.1, 7.9, 8.2, 7.9, 7.5],
                                        [13.9, 12.7, 11.5, 11.1,
                                            11.7, 11.5, 11.5, 10.3],
                                        [20.3, 18.4, 16.9, 17.2,
                                            16.7, 16.4, 15.6, 14.8],
                                        [29.2, 27.6, 25.8, 25.6,
                                            25.6, 25.5, 23.6, 23.6],
                                        [36.8, 33.8, 33.3, 33.8,
                                            34.5, 34.5, 33.7, 32.9],
                                        [73.3, 75.3, 73.6, 72.1, 73.2, 71.7, 68.2, 67.0]]) / 1000

        self.mortality_female = np.array([[1.4, 1.2, 1.2, 1.2, 1.2, 1.0, 0.9, 0.7],  # смертность женщин на 1000 человек соответствующего возраста (по 5 лет)
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
                                          [11.0, 10.8, 10.6, 9.7,
                                           10.2, 9.3, 9.2, 9.0],
                                          [14.8, 15.1, 14.6, 15.0,
                                           15.0, 15.2, 14.6, 13.7],
                                          [59.0, 61.4, 61.3, 62.1, 63.9, 62.3, 60.7, 57.8]]) / 1000

        self.rate_people = np.array([[4.2, 4.5, 4.9, 5.2, 5.6, 6.0, 6.3, 6.2],  # процент (доля) людей соответствующего возраста в популяции (по 5 лет)
                                     [3.8, 3.8, 3.8, 3.8, 3.9, 4.0, 4.3, 4.7],
                                     [3.4, 3.4, 3.4, 3.5, 3.6, 3.7, 3.7, 3.7],
                                     [5.1, 4.5, 4.0, 3.9, 3.7, 3.7, 3.8, 3.9],
                                     [8.8, 8.7, 8.3, 7.7, 6.9, 6.0, 5.2, 4.8],
                                     [8.8, 9.1, 9.4, 9.6, 9.7, 9.6, 9.3, 8.7],
                                     [8.0, 8.2, 8.4, 8.8, 9.0, 9.1, 9.3, 9.6],
                                     [7.4, 7.4, 7.5, 7.6, 7.7, 7.9, 8.0, 8.3],
                                     [6.7, 6.8, 7.0, 7.1, 7.2, 7.1, 7.2, 7.2],
                                     [7.5, 7.1, 6.7, 6.5, 6.4, 6.4, 6.5, 6.6],
                                     [8.0, 8.0, 7.8, 7.6, 7.4, 7.0, 6.6, 6.3],
                                     [7.1, 7.2, 7.2, 7.2, 7.2, 7.3, 7.3, 7.2],
                                     [6.5, 6.4, 6.3, 6.2, 6.2, 6.3, 6.4, 6.4],
                                     [3.1, 3.4, 4.3, 4.8, 5.3, 5.6, 5.5, 5.5],
                                     [11.6, 11.5, 11.0, 10.5, 10.2, 10.3, 10.6, 10.9]]) / 100
        self.fertility = np.array([[11.9, 12.3, 11.9, 11.8, 11.3, 9.7, 8.2, 7.6],  # количество рожавших на 1000 женщин соответствующего возраста (по 5 лет с 15-ти)
                                   [53.0, 52.8, 50.4, 49.2, 51.1, 51.9, 48.2, 49.3],
                                   [98.7, 104.4, 102.9, 102.4,
                                       106.4, 104.9, 92.7, 87.0],
                                   [73.3, 81.6, 83.5, 89.2,
                                       94.4, 100.2, 91.7, 88.9],
                                   [34.7, 40.2, 42.2, 45.5, 47.9, 51.9, 49.8, 50.0],
                                   [7.3, 8.9, 9.0, 10.2, 11.3, 12.3, 12.3, 12.7],
                                   [0.5, 0.4, 0.6, 0.7, 0.7, 1.0, 1.0, 1.1]]) / 1000
        self.birth_sex = np.array([[29.5, 32.3, 33.2, 34.6, 36.5, 37.3, 34.3, 32.9],  # количество родившихся мальчиков/девочек по годам
                                   [27.5, 30.4, 31.0, 32.6, 34.2, 35.4, 32.2, 31.1]]) * 1000
        self.migration = np.array([[130321, 194511, 257636, 261972, 237084, 232663, 264780, 250742],  # реальное количество мигрировавших (им/э) по годам
                                   [71689, 120419, 157619, 209176, 211821, 187954, 200234, 222966]])
        self.migration_male = np.array([[4320, 6649, 4208, 6382, 4680, 7533, 3504, 8122],  # количество мигрировавших мужчин (3+/3-) соответствующего возраста
                                        [4270, 7335, 4159, 7510,
                                            3594, 6631, 2691, 7463],
                                        [2845, 4103, 2772, 4164,
                                            2219, 3249, 1662, 3731],
                                        [14226, 14193, 13857, 13562,
                                            10205, 6145, 7642, 6916],
                                        [23600, 15622, 24883, 14721,
                                            20758, 13733, 13467, 17461],
                                        [22702, 15935, 23910, 14223,
                                            18520, 10917, 12978, 12283],
                                        [17565, 16573, 17783, 15475,
                                            14266, 12193, 10584, 13478],
                                        [12719, 11778, 12888, 11215,
                                            10885, 9101, 8075, 10325],
                                        [10150, 8212, 10891, 8035,
                                            8894, 6512, 6904, 7196],
                                        [8287, 6491, 8386, 6187,
                                            7377, 5352, 5726, 5779],
                                        [6273, 5227, 5864, 4849,
                                            5505, 4352, 4354, 4637],
                                        [4220, 4647, 3615, 4465,
                                            3561, 3748, 2803, 3971],
                                        [2488, 3224, 2191, 3153,
                                            2150, 2805, 1944, 2937],
                                        [1452, 2030, 1264, 2057,
                                            1402, 1744, 1226, 1762],
                                        [717, 964, 621, 1011, 605, 750, 530, 923],
                                        [673, 819, 400, 651, 560, 654, 490, 525],
                                        [482, 698, 444, 723, 332, 506, 290, 525]])
        self.migration_female = np.array([[4048, 6063, 3769, 6051, 4363, 6851, 3227, 7690],  # количество мигрировавших женщин (3+/3-) соответствующего возраста
                                          [4046, 6995, 3767, 6880,
                                              3298, 6210, 2440, 6848],
                                          [2946, 3906, 2743, 3851,
                                              2127, 2989, 1574, 3400],
                                          [12288, 14314, 11440, 13854,
                                              7683, 5535, 5683, 6065],
                                          [17582, 15546, 17390, 14526,
                                           12177, 14060, 9083, 16711],
                                          [20858, 19321, 20715, 16588,
                                           14554, 12975, 11112, 13553],
                                          [15725, 19370, 15046, 18318,
                                           11754, 14482, 8783, 15840],
                                          [10794, 12731, 10328, 12478,
                                              8720, 9935, 6516, 11112],
                                          [8088, 8647, 8015, 8259,
                                              6428, 6458, 5006, 7326],
                                          [6063, 6605, 6009, 6184,
                                              4957, 4855, 3860, 5501],
                                          [5988, 6057, 5411, 5499,
                                              4622, 4407, 3894, 4544],
                                          [4955, 6131, 4267, 5870,
                                              3973, 4609, 3355, 4898],
                                          [3695, 4721, 3186, 4573,
                                              2984, 3566, 2594, 3942],
                                          [2488, 3483, 2233, 3468,
                                              2034, 2706, 1899, 2770],
                                          [1598, 1789, 1435, 1947,
                                              1097, 1354, 1024, 1527],
                                          [1868, 2145, 1677, 1708,
                                              1389, 1590, 1297, 1355],
                                          [1953, 2456, 2069, 2305, 1503, 1727, 1403, 1850]])
        self.data = pd.DataFrame({})
        self.number_of_age_classes = 16

    # @staticmethod
    def factor_age(self, age):
        return (age//self.age_step) if age <= 80 else self.number_of_age_classes

    def lagrange_new(self, x_interp: int, x_data: List[int], y_data: List[float], n=2):
        y_interp = 0
        for i in range(n+1):
            multiplication = 1
            for j in range(n+1):
                if (i != j):
                    multiplication *= (x_interp -
                                       x_data[j])/(x_data[i] - x_data[j])
            y_interp += (y_data[i]*multiplication)
        return y_interp

    def read_data(self, filename):
        self.data = pd.read_csv(filename, sep='\t', index_col=0)
        self.data = self.data[['sp_id', 'sp_hh_id', 'age', 'sex']]
        self.data[['sp_id', 'sp_hh_id', 'age']] = self.data[[
            'sp_id', 'sp_hh_id', 'age']].astype(int)
        self.data[['sex']] = self.data[['sex']].astype(str)
        self.data['factor_age'] = list(
            map(lambda age: self.factor_age(age), self.data.age))

    def update_age(self):
        self.data.age += 1

    # remove deceased from data
    def population_after_mortality(self):
        age_max = 100
        total_year_mortality = 0
        mortality_indexes_of_all_ages = []
        for i in range(1, age_max):
            current_age_people = self.data.loc[self.data['age'] == i]
            males_at_current_age = current_age_people.loc[current_age_people['sex'] == 'M']
            # display(males_at_current_age)
            females_at_current_age = current_age_people.loc[current_age_people['sex'] == 'F']
            male_mortality = len(males_at_current_age) * \
                self.mortality_male_interpolated[i]
            female_mortality = len(females_at_current_age) * \
                self.mortality_female_interpolated[i]
            total_year_mortality += (male_mortality + female_mortality)
            print('Male: \n Age: {}, Mortality: {}, People at current age: {}'.format(
                i,  male_mortality, males_at_current_age.shape[0]))
            print('Female: \n Age: {}, Mortality: {}, People at current age: {}'.format(
                i,  female_mortality, females_at_current_age.shape[0]))
            mortality_male_indexes = np.random.choice(np.array(males_at_current_age.sp_id),
                                                      int(round(male_mortality)),
                                                      replace=False)
            mortality_female_indexes = np.random.choice(np.array(females_at_current_age.sp_id),
                                                        int(round(
                                                            female_mortality)),
                                                        replace=False)
            mortality_indexes_of_all_ages.extend(
                mortality_male_indexes)
            mortality_indexes_of_all_ages.extend(mortality_female_indexes)

        print(Counter(self.data.loc[self.data.sp_id.isin(
            mortality_indexes_of_all_ages), 'factor_age']))
        self.data = self.data[~self.data.sp_id.isin(
            mortality_indexes_of_all_ages)]

    def plot_mortality(self):
        age = [5*i + 2 for i in range(len(self.mortality_male))]
        mortality = self.mortality_male[:, 0]
        plt.plot(age, mortality, 'o')
        plt.show()

    def plot_fertility(self):
        age = [15 + 5*i + 2 for i in range(len(self.fertility))]
        fertility = self.fertility[:, 0]
        plt.plot(age, fertility, 'o')
        plt.show()

    # returns mortality per 1000 people in each age from 1 to 100
    def interpolate_mortality(self, sex, year) -> List[int]:
        if sex == 'M':
            mortality = self.mortality_male[:, year - 2011]
        elif sex == 'F':
            mortality = self.mortality_female[:, year - 2011]

        age = [self.age_step*i + 2 for i in range(len(mortality))]
        age_min = 1
        age_max = 100

        mortality_interpolated = []
        # interpolate by data between given ages
        for i in range(len(mortality)-3):
            for j in range(self.age_step*i, self.age_step*i+self.age_step):
                mortality_interpolated.append(
                    self.lagrange_new(j, age[i:i+3], mortality[i:i+3]))
        # extrapolate values to age_max
        for j in range(age[-3], age_max+2):
            interpolated_value = self.lagrange_new(j, age[-3:], mortality[-3:])
            if interpolated_value < 1:
                mortality_interpolated.append(interpolated_value)
            else:
                mortality_interpolated.append(1)

        age_new = np.linspace(1, len(mortality_interpolated),
                              len(mortality_interpolated))
        plt.scatter(age, mortality, label='data')
        plt.scatter(age_new, mortality_interpolated,
                    label='Polynomial', alpha=0.2)
        plt.grid(True)
        plt.show()
        if sex == 'M':
            self.mortality_male_interpolated = mortality_interpolated
        elif sex == 'F':
            self.mortality_female_interpolated = mortality_interpolated
        return mortality_interpolated

    def interpolate_fertility(self, year) -> List[int]:
        fertility = self.fertility[:, year - 2011]
        age = [5*i + 1 for i in range(3, 3 + len(fertility))]
        age_min = 1
        age_max = 50

        fertility_interpolated = []
        # interpolate by data between given ages
        for i in range(len(fertility)-3):
            for j in range(15 + 5*i, 15 + 5*i+5):
                fertility_interpolated.append(
                    self.lagrange_new(j, age[i:i+3], fertility[i:i+3]))
        # extrapolate values to age_max
        for j in range(age[-3], age_max+1):
            fertility_interpolated.append(
                self.lagrange_new(j, age[-3:], fertility[-3:])/self.age_step)

        age_new = np.linspace(15, 15 + len(fertility_interpolated),
                              len(fertility_interpolated))
        plt.scatter(age, fertility, label='data')
        plt.scatter(age_new, fertility_interpolated,
                    label='Polynomial', alpha=0.2)
        plt.grid(True)
        plt.show()
        return fertility_interpolated

    def display_data(self):
        display(self.data)


FPOP_1 = FPOP()

mort_array = []
FPOP_1.display_data()
year = 2011
FPOP_1.read_data(r'people_{}.txt'.format(year))
FPOP_1.interpolate_mortality('M', year)
FPOP_1.interpolate_mortality('F', year)
mort_array.append(FPOP_1.population_after_mortality())
FPOP_1.display_data()

print(mort_array)

# year = 2011
# FPOP_1.read_data(r'people_{}.txt'.format(year))
# FPOP_1.interpolate_fertility(year)
