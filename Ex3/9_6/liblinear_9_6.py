from liblinear.python.liblinear import problem
from liblinear.python.commonutil import *
import math
# from liblinear.python.liblinearutil import train

def preData(filename: str):
    y, x = svm_read_problem(filename)
    for line in x:
        for key in line.keys():
            line[key] = line[key] ** 0.5
    file = open(filename + "_pre", "a")
    for i in range(0, len(x)):
        output = ""
        output += str(int(y[i]))
        line = x[i]
        for key in line.keys():
            output += " "
            output += str(key)
            output += ":"
            output += str(line[key])
        output += "\n"
        file.write(output)
    file.close()


# preData("mnist")
# preData("mnist.t")

preData("mnist.scale.t")
preData("mnist.scale")