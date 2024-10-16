#SI-model of Baroyan-Rvachev origin with discrete time
#v2 - using numpy.optimize instead of iterating through k
#v3 - including I0 into fitted params
#v4 - considering bias in t_peak (iterating through possible value range)
#v5 - prediction on partial data
#v6 - fixed algorithm, flexible number of init seeds
#v7 - age structure (0-2, 3–6, 7–14, and 15+ years old), no calibration
#v8 - fitting for weekly data, no holidays removal
#BR_model - the script version in a form of a class
#v2 - new launch procedure (allows different lambda, alpha for different age groups)
#v3 - no 'total'
#_strains - a version with strains, no age structure




import numpy as np
import data_functions as datf

import matplotlib.pyplot as plt

def f(h, m, a):
    if h==m: #The individual was exposed to the same virus strain as in the previous season
        return a
    else:
        return 1


class BR_model:
    #age-sctructured Baroyan-Rvachev model with comparison for the averaged model without age groups
    q = []  # infectivity for a fixed infected period

    strains = ["A(H1N1)pdm09", "A(H3N2)", "B"]
    history_states = ["A(H1N1)pdm09", "A(H3N2)", "B", "No exposure"]
    strains_num = 3
    history_states_num = 4

    exposed_fraction_h = [] #fractions with different exposure history
    rho = 0 #A scalar in absence of separate age groups
    lam_m = []
    cont_num = 6.528
    Phi = []
    a = 0 #Waning immunity level

    total_recovered = [] #Вылечившиеся от разных штаммов

    N = 0  # 00 #modeling interval

    def __init__(self, *args):
        pass

    @staticmethod
    def init(q, pop_size, Phi, N):
        model = BR_model()
        model.q = q
        model.rho = pop_size
        model.Phi = Phi
        model.N = N
        
        return model

    def initSimulParams(self, exposed_list, lam_list, a):
        self.exposed_fraction_h = exposed_list.copy()
        self.lam_m = lam_list.copy()
        self.a = a


    def sum_ill (self, y, t, phi):
    #summing the cumulative infectivity of the infected by the strain m at the moment t
        sum = 0
        T = len(self.q)

        for epid_day in range(0, T):
            if t-epid_day<=0:  #phi[0], phi[1],...
                if epid_day-t < phi.size:
                    y_cur = phi[epid_day-t]
                else:
                    y_cur = 0
            else: #t=0 отправили в предысторию
                y_cur = y[t-epid_day] #y[1], y[2],...

            sum = sum + y_cur*self.q[epid_day]
        return sum

    @staticmethod
    def calcRelativePrevalence(y_model, rho):
        # y_model is a 2D array
        # Converts the absolute prevalence into relative (per 10000 persons)
        y_rel = []
        for i in range(0, len(y_model)):
            y_rel.append(datf.real_to_abs(y_model[i], rho))

        return y_rel


    def MakeSimulation (self):

        magic_prehistory_length = 10

        y = np.zeros((self.strains_num,self.N+1))
        x = np.zeros((self.history_states_num,self.N+1))

        self.total_recovered = [0, 0, 0] #Обнуляется список

        for h in range(0, self.history_states_num):  # initial data
            #print("Init")
            x[h][0] = self.exposed_fraction_h[h]*self.rho
            #print(x[h][0])


        for t in range(0,self.N): #Стартовый момент t+1 = 1

            for m in range(0, self.strains_num):  # calculating y_m
                y[m][t + 1] = 0

            for h in range(0, self.history_states_num):
                #print("")
                #print("H: ", h)
                x[h][t + 1] = x[h][t]

                infect_force_total = 0 #Инфекция от всех штаммов на группу h
                infect_force = []
                for m in range(0, self.strains_num):  # calculating y_m
                    #y[m][t + 1] = 0

                    infect_force.append(self.lam_m[m] * self.cont_num * self.sum_ill(y[m], t, self.Phi[m]) * f(h,m,self.a) / self.rho)

                    #print("{}: f: {} Force m {}".format(m, f(h,m,self.a), infect_force[m]))
                    infect_force_total+= infect_force[m] #Считаем общую силу инфекции

                real_infected = min(infect_force_total, 1.0) * x[h][t]
                #print("Total inf:", real_infected)

                #print(infect_force_total)

                x[h][t + 1] -= real_infected

                if infect_force_total > 0:
                    for m in range(0, self.strains_num):  # calculating y_m
                        real_infected_m = real_infected * (infect_force[m] / infect_force_total) # Причитающаяся доля
                        y[m][t + 1] += real_infected_m
                        self.total_recovered[m] += real_infected_m  # Они переболеют (нет смертности)

        return y

