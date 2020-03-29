# 用于测试的数据
testData = [
    [1, 1, 1.0],
    [2, 2, 0.9],
    [3, 1, 0.8],
    [4, 1, 0.7],
    [5, 2, 0.6],
    [6, 1, 0.5],
    [7, 2, 0.4],
    [8, 2, 0.3],
    [10, 2, 0.1],
    [9, 1, 0.2]
    # 这里故意引入倒叙，用来测试排序函数
]


def takeThird(item: list):
    """
    :param item: 列表
    :return: 列表的第三个元素
    用于后续的排序调用，按照第三个元素，也就是预测值进行排序
    """
    return item[2]


def sortData(data: list) -> list:
    data.sort(key=takeThird, reverse=True)  # 依照预测值，对列表进行降序排序
    # print(data)
    return data


testData = sortData(testData)

pos_N = 0
neg_N = 0   # 分别用来记录正类和反类的样本数
for item in testData:
    if item[1] == 1:
        pos_N += 1
    else:
        neg_N += 1

P = [1.0000]
R = [0.0000]  # 分别用来记录查准率Precision和查全率Recall的列表

true_pos_n = 0   # 用来记录当前真正类的个数
for i in range(0, len(testData)):
    if testData[i][1] == 1:
        true_pos_n += 1
    P.append(true_pos_n / (i + 1))
    R.append(true_pos_n / pos_N)

print("查准率为：", P)
print("查全率为：", R)

AUC_PR = []
AP = []     # 字面意思，用来记录的列表

for i in range(1, len(testData) + 1):
    AUC_PR.append((R[i] - R[i-1]) * (P[i] + P[i-1]) / 2)
    AP.append((R[i] - R[i-1]) * P[i])

print("AUC-PR为：", AUC_PR, "和为%f" % sum(AUC_PR))
print("AP为：", AP, "和为%f" % sum(AP))