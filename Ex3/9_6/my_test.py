from liblinear.python.liblinear import problem
from liblinear.python.commonutil import *
import math
# from liblinear.python.liblinearutil import train

def findData(filename: str):
    max_data = dict()
    y, x = svm_read_problem(filename)
    for i in range(0, len(x)):
        line = x[i]
        for key in line.keys():
            if key in max_data.keys():
                max_data[key] = max(line[key], max_data[key])
            else:
                max_data[key] = line[key]
    print(max_data)


findData("mnist")
