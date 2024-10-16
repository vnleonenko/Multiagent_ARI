import csv
import numpy as np

#Functions for handling data

#!!!!!!!!!!!!!!!!!!!!!!!
#DEPRECATED FOR BAROYANAGE!!!!
#But some functions are still in use
#!!!!!!!!!!!!!!!!!!!!!!!

#TODO: remove from the project moving time_string to idatf

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)

def readFromCsvToList(filename):
    # return list with all data from csv file, skipping the first row (headers)
    reader = csv.reader(open(filename), delimiter=';')
    next(reader)
    res_list = list(reader)
    return res_list


def generateNumSequenceFromZero(final):
#generates the sequence of the form 0, +-1, +-2, ..., +-final
    num_list = [0]

    for i in range(1,final):
        num_list.extend([i, -i])

    return np.array(num_list)


def doesContainEpidPeak( inc_data ):
    prev_inc = 0
    for cur_inc in inc_data:
        if cur_inc<prev_inc:
            return True #We have a peak in data (considering there's no minor peaks before the big one)
        prev_inc = cur_inc

    return False #lacking data on incidence peaks

def max_elem_index( my_list ):
#returns the index of a highest incidence
    max_value = max(my_list)
    max_index = my_list.index(max_value)
    return max_index

def max_elem_indices( df, keys_list ):
#returns the index of a highest incidence
    max_values_list = []
    max_indices_list = []
    for key in keys_list:
        my_list = list(df[key])
        #print(my_list)
        max_value = max(my_list)
        max_values_list.append(max_value)
        max_indices_list.append(my_list.index(max_value))
    return max_indices_list, max_values_list

def max_peak_index( df, keys_list ):
#returns the index of a highest incidence

    max_values_list = []
    for key in keys_list:
        my_list = list(df[key])
        #print(my_list)
        max_values_list.append(max(my_list))

    peak_index = max_values_list.index(max(max_values_list)) #Finding the highest incidence among all the strains

    return peak_index

def calculate_dist_squared(x, y, delta):
#calculating the fitting coefficient r
#x is real data, y is modeled curve
    #delta is the difference between the epidemic starts in real data and modeled curve
    sum = 0
    for i in range(delta,delta+len(x)):
        #if x[i-delta]>0 and y[i]>0: #do not consider absent data which is marked by -1
        sum = sum + pow(x[i-delta] - y[i], 2)

    return sum

def calculate_dist_squared_list(df_data, df_simul, strains, delta):
#x is real data, y is modeled curve
    #delta is the difference between the epidemic starts in real data and modeled curve

    sum_list = []
    for strain in strains:
        x = list(df_data[strain])
        y = list(df_simul[strain])

        sum = 0
        for i in range(delta,delta+len(x)):
            sum = sum + pow(x[i-delta] - y[i], 2)

        sum_list.append(sum)

    return sum_list

def calculate_dist_squared_weighted(x, y, delta, w):
#calculating the fitting coefficient r
#x is real data, y is modeled curve
#delta is the difference between the epidemic starts in real data and modeled curve
#w are the weights marking the importance of particular data points fitting

    sum = 0
    for i in range(delta,delta+len(x)):
        #if x[i-delta]>0 and y[i]>0: #do not consider absent data which is marked by -1
        sum = sum + w[i-delta]*pow(x[i-delta] - y[i], 2)

    return sum


def calculate_dist_squared_weighted_list(df_data, df_simul, strains, delta, w):
#x is real data, y is modeled curve
    #delta is the difference between the epidemic starts in real data and modeled curve

    sum_list = []
    for strain in strains:
        x = list(df_data[strain])
        y = list(df_simul[strain])

        sum = 0
        for i in range(delta,delta+len(x)):
            sum = sum + w[strain][i-delta]*pow(x[i-delta] - y[i], 2)

        sum_list.append(sum)

    return sum_list


def calculate_peak_bias(x, y):
    x_peak = max(x)
    #print("Real peak height:", x_peak)
    y_peak = max(y)
    #print("Model peak height:", y_peak)
    #return abs(x_peak-y_peak)
    return y_peak/x_peak

def find_residuals( data ):
    res = 0
    mean = np.mean(data)
    for i in range(0, len(data)):
        res+=pow(data[i] - mean, 2)
    return res

def find_residuals_list( df, strains ):
    res_list = []
    for strain in strains:
        res = 0
        data = list(df[strain])
        mean = np.mean(data)
        for i in range(0, len(data)):
            res+=pow(data[i] - mean, 2)
        res_list.append(res)
    return res_list

def find_residuals_weighted( data, w ):
    res = 0
    mean = np.mean(data)
    for i in range(0, len(data)):
        res+=w[i]*pow(data[i] - mean, 2)
    return res

def find_residuals_weighted_list( df, strains, w ):
    res_list = []
    for strain in strains:
        res = 0
        data = list(df[strain])
        mean = np.mean(data)
        for i in range(0, len(data)):
            res+=w[strain][i]*pow(data[i] - mean, 2)
        res_list.append(res)
    return res_list

def real_to_abs (array, rho):
    array_out = []
    for t in range(0, len(array)):
        array_out.append(array[t] * 10000.0 / float(rho))
    return np.array(array_out)

