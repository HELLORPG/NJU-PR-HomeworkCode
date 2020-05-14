import math


function = [
    [1/2, 1/2],
    [1/4, 3/4],
    [1/8, 7/8]
]

KL = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

def ComputeKL(p: int, q: int, f: list) -> float:
    return f[p][0] * math.log(f[p][0] / f[q][0], 2) + f[p][1] * math.log(f[p][1] / f[q][1], 2)


def computeAllKL():
    global function, KL
    for i in range(0, 3):
        for j in range(0, 3):
            KL[i][j] = ComputeKL(i, j, function)
    return


def geqZeroTest():
    for line in KL:
        for data in line:
            if data < 0:
                print("KL[%d][%d] less than 0." % (KL.index(line), line.index(data)))
                return
    print("ALL KL are geq 0.")
    return


def symmetryTest():
    for i in range(0, 3):
        for j in range(i, 3):
            if KL[i][j] != KL[j][i]:
                print("KL[%d][%d] neq KL[%d][%d], not symmetry." % (i, j, j, i))
                return
    print("KL is symmetry!")
    return


def triangleTest():
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                if KL[i][j] + KL[j][k] < KL[i][k]:
                    print("KL[%d][%d] + KL[%d][%d] < KL[%d][%d], not fit triangle inequality." % (i, j, j, k, i, k))
                    return
    print("KL fits triangle inequality!")
    return


def main():
    computeAllKL()
    print(KL)
    geqZeroTest()
    symmetryTest()
    triangleTest()


main()