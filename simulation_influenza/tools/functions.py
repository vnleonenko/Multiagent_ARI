import time
import multiprocessing as mp
from functools import wraps

def file_timer(func):
    ''' сохраняет время выполнения функции в файл '''

    @wraps(func)
    def inner(*args, **kwargs):
        
        start_all = time.perf_counter()

        res = func(*args, **kwargs)

        time.perf_counter() - start_all

        with open(args[0].results_dir + 'time.txt', 'w') as f:
            print(time.perf_counter() - start_all, file=f)

        return res

    return inner


def day_timer(func):
    ''' выводит время выполнения функции в консоль '''

    @wraps(func)
    def inner(*args, **kwargs):

        start = time.perf_counter()

        res = func(*args, **kwargs)

        print("Proc {}, Day {}, time elapsed: {} sec\n".format(mp.current_process().name, args[1], time.perf_counter() - start))

        return res

    return inner

