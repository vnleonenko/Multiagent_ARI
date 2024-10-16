import BR_model_strains_hybrid as mclass
import numpy as np


import matplotlib
import matplotlib.pyplot as plt

strains = ["A(H1N1)pdm09", "A(H3N2)", "B"]
STRAINS_NUM = 3

def plotStrainsDynamics(y, strains_list, t0=0):

    colors = ['r', 'b', 'g', 'm']

    N = len(y[0])

    fig = plt.figure(figsize=(10, 6))
    matplotlib.rcParams.update({'font.size': 14})

    for i in range(0, STRAINS_NUM):
        plt.plot(range(t0,t0+N), y[i][:N], "--", color = colors[i], label=strains_list[i], linewidth=3.0)


    plt.legend(loc='best')#, fancybox=True, shadow=True)


    plt.ylabel('ARI incidence, cases') #Relative ARI incidence, cases per 10000 persons

    plt.title('')
    plt.grid()

    plt.show()
    plt.close()

    #plt.savefig(fname, dpi=150, bbox_inches='tight')



def sumQuantities(y):
    y_sum = []
    for j in range(len(y[0])):
        y_sum.append(np.sum([y[i][j] for i in range(len(y))]) )


    return y_sum

# def Test():
#
#     pop_num = 4800000 #TODO: download from files
#     N = 60
#     q = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]
#
#     exposure_list = [0.0, 0.0, 0.0, 1.0] #Sum=1
#
#     lam_list = [0.3, 0.3, 0.3]
#
#     a=0.7
#
#     I0 = [1,1,1]
#
#     model = mclass.BR_model.init(q, pop_num, I0, N)
#     model.initSimulParams(exposure_list, lam_list, a)
#     y_model = model.MakeSimulation()
#     y_model_sum = sumQuantities(y_model)
#     y_rel = mclass.BR_model.calcRelativePrevalence(y_model, pop_num)
#
#     plotStrainsDynamics(y_model, strains)
#     #plotStrainsDynamics(y_rel, strains)


def createExposedList(immune_list):
    sum_exposed = immune_list[0] + immune_list[1] + immune_list[2]
    if sum_exposed <= 1:  # Summ of exposed less than 100%
        percent_naive = 1 - immune_list[0] - immune_list[1] - immune_list[2]  # 4 state, naive to infection

    else:  # normalization
        percent_naive = 0
        immune_list[0] = immune_list[0] / sum_exposed
        immune_list[1] = immune_list[1] / sum_exposed
        immune_list[2] = immune_list[2] / sum_exposed

    return [immune_list[0], immune_list[1], immune_list[2], percent_naive]

def launchFromABM(Phi, Recovered_num, Immune_list, q, lmbd, pop_num, percent_protected, t0, N):
    #percent_protected - полностью защищённые
    exposure_list = createExposedList(Immune_list) #Изначально иммунные по прошлой заболеваемости

    lam_list = [lmbd]*len(strains)

    a=0.3

    pop_num_S = max(pop_num*(1-percent_protected) - Recovered_num, 0)

    model = mclass.BR_model.init(q, pop_num_S, Phi, N)
    model.initSimulParams(exposure_list, lam_list, a)
    y_model = model.MakeSimulation()
    # y_model_sum = sumQuantities(y_model)
    # y_rel = mclass.BR_model.calcRelativePrevalence(y_model, pop_num)

    #plotStrainsDynamics(y_model, strains, t0)
    return y_model


#Test()

