"""
此代码用于寻找相同类型订单的分布特性，包括分布类型、相关参数等
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter

sns.set(style="darkgrid")

choose = 9

if choose == 1:
    # 平均准确率为81.37%
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
         '12', '13', '14', '15', '16', '17', '18', '19', '20']
    #     1     2      3     4    5     6     7     8     9     10
    y = [81.4, 80.3, 82.1, 81.2, 83.0, 84.2, 80.7, 81.6, 83.7, 83.0,
    #     11    12    13    14    15    16    17    18    19   20
         79.7, 81.2, 83.0, 81.4, 78.8, 80.4, 84.0, 82.1, 78.0, 77.6]
    sns.barplot(x=x, y=y, order=x)
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("label: day")
    plt.ylabel("accuracy rate: %")
    plt.title("Based on the distribution")
    plt.show()

elif choose == 2:
    # 平均准确率为71.29%
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
         '12', '13', '14', '15', '16', '17', '18', '19', '20']
    #     1     2      3     4    5     6     7     8     9     10
    y = [72.1, 71.9, 72.8, 73.0, 73.2, 70.2, 71.4, 72.4, 69.8, 74.5,
         #     11    12    13    14    15    16    17    18    19   20
         70.0, 71.8, 73.6, 70.4, 69.8, 68.6, 70.1, 72.2, 68.2, 69.8]
    sns.barplot(x=x, y=y, order=x)
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("label: day")
    plt.ylabel("accuracy rate: %")
    plt.title("The original method")
    plt.show()

elif choose == 3:
    x = ['normal', 'exponential', 'gamma', 'chi', 'F', 'other']
    y = [82.6, 80.3, 83.4, 81.6, 80.8, 79.4]
    sns.barplot(x=x, y=y, order=x)
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("distribution")
    plt.ylabel("accuracy rate: %")
    plt.title("The accuracy of different distribution")
    plt.show()

elif choose == 4:
    x = ['linear', 'svm', 'dt', 'gbdt', 'rf', 'erf', 'dis-erf']
    y = [65.3, 67.8, 69.2, 69.6, 71.2, 73.4, 81.3]
    sns.barplot(x=x, y=y, order=x)
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("model name")
    plt.ylabel("accuracy rate: %")
    plt.title("different models")
    plt.show()

elif choose == 5:
    x = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
    y = [75.6, 76.1, 78.2, 79.2, 81.5, 81.2, 81.0, 80.8, 80.1]
    sns.lineplot(x=x, y=y)
    plt.yticks(np.arange(74, 85, 1))
    plt.xlabel("n_estimator")
    plt.ylabel("accuracy rate: %")
    plt.title("different estimator")
    plt.show()

elif choose == 6:
    x = ['10', '20', '30', '40', '50', '60', '70', 'none']
    y = [78.3, 79.4, 80.1, 81.2, 81.0, 80.8, 80.0, 80.2]
    sns.lineplot(x=x, y=y)
    plt.yticks(np.arange(75, 85, 1))
    plt.xlabel("max_depth")
    plt.ylabel("accuracy rate: %")
    plt.title("different max_depth")
    plt.show()

elif choose == 7:
    # 平均准确率为81.37%
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
         '12', '13', '14', '15', '16', '17', '18', '19', '20']
    #     1     2      3     4    5     6     7     8     9     10
    y = [80.4, 79.3, 81.1, 80.2, 82.0, 83.2, 79.7, 80.6, 82.7, 82.0,
    #     11    12    13    14    15    16    17    18    19   20
         78.7, 80.2, 82.0, 80.4, 77.8, 79.4, 82.0, 81.1, 77.0, 76.6]
    z = [72.1, 71.9, 72.8, 73.0, 73.2, 70.2, 71.4, 72.4, 69.8, 74.5,
         #     11    12    13    14    15    16    17    18    19   20
         70.0, 71.8, 73.6, 70.4, 69.8, 68.6, 70.1, 72.2, 68.2, 69.8]
    bar_width = 0.3
    x_1 = list(range(len(x)))
    x_2 = [i + bar_width for i in x_1]
    x_3 = [i + bar_width * 0.5 for i in x_1]
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    plt.bar(x_1, y, width=bar_width, label="dis-ert")
    plt.bar(x_2, z, width=bar_width, label="ert")
    plt.legend()
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("label: day")
    plt.ylabel("accuracy rate: %")
    plt.xticks(x_1, x)
    plt.title("Based on the distribution")
    plt.show()

elif choose == 8:
    # 平均准确率为81.37%
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
         '12', '13', '14', '15', '16', '17', '18', '19', '20']
    #     1     2      3     4    5     6     7     8     9     10
    y = [80.4, 79.3, 81.1, 80.2, 81.0, 82.2, 78.7, 80.6, 81.7, 81.0,
    #     11    12    13    14    15    16    17    18    19   20
         78.7, 80.2, 81.0, 80.4, 76.8, 78.4, 82.0, 80.1, 76.0, 75.6]
    z = [72.1, 71.9, 72.8, 73.0, 73.2, 70.2, 71.4, 72.4, 69.8, 74.5,
         #     11    12    13    14    15    16    17    18    19   20
         70.0, 71.8, 73.6, 70.4, 69.8, 68.6, 70.1, 72.2, 68.2, 69.8]
    bar_width = 0.3
    x_1 = list(range(len(x)))
    x_2 = [i + bar_width for i in x_1]
    x_3 = [i + bar_width * 0.5 for i in x_1]
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    plt.bar(x_1, y, width=bar_width, label="cascade-ert")
    plt.bar(x_2, z, width=bar_width, label="ert")
    plt.legend()
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("label: day")
    plt.ylabel("accuracy rate: %")
    plt.xticks(x_3, x)
    plt.show()

elif choose == 9:
    # 平均准确率为81.37%
    x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
         '12', '13', '14', '15', '16', '17', '18', '19', '20']
    #     1     2      3     4    5     6     7     8     9     10
    y = [80.4, 78.3, 80.1, 80.2, 81.0, 81.2, 77.7, 80.6, 80.7, 80.0,
    #     11    12    13    14    15    16    17    18    19   20
         77.7, 78.2, 81.0, 80.4, 76.8, 78.4, 79.0, 80.1, 76.0, 75.6]
    z = [81.4, 80.3, 82.1, 81.2, 83.0, 84.2, 80.7, 81.6, 83.7, 83.0,
    #     11    12    13    14    15    16    17    18    19   20
         79.7, 81.2, 83.0, 81.4, 78.8, 80.4, 84.0, 82.1, 78.0, 77.6]
    bar_width = 0.3
    x_1 = list(range(len(x)))
    x_2 = [i + bar_width for i in x_1]
    x_3 = [i + bar_width * 0.5 for i in x_1]
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.grid(False)
    plt.bar(x_2, y, width=bar_width, label="dis-cascade-ert")
    plt.bar(x_1, z, width=bar_width, label="cascade-ert")
    plt.legend()
    plt.ylim([70, 85])
    # plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("label: day")
    plt.ylabel("accuracy rate: %")
    plt.xticks(x_3, x)
    plt.show()

elif choose == 10:
    x = ['1', '2', '3', '4', '5']
    y = [76.6, 81.4, 81.0, 81.2, 81.0]
    sns.lineplot(x=x, y=y)
    plt.yticks(np.arange(70, 85, 1))
    plt.xlabel("layers")
    plt.ylabel("accuracy rate: %")
    plt.title("different layers")
    plt.show()

