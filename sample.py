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


# origin_data = pd.read_csv('../data/d_origin_data.csv')
# data = stats.gamma.rvs(2, loc=1.5, scale=2, size=100000)
# f = Fitter(data)
# f.fit()
# f.summary()
# f.hist()


# def data_split_by_wgbez():
#     wgbez_set = set(origin_data['wgbez'].values)
#     wgbez_dict = dict()
#     for w in wgbez_set:
#         wgbez_dict[w] = pd.DataFrame()
#         wgbez_index = origin_data[origin_data['wgbez'] == w].index
#         wgbez_dict[w] = origin_data.loc[wgbez_index]
#     return wgbez_set, wgbez_dict
#
#
# def data_split_by_date():
#     date_set = set(origin_data['actual_day'].values)
#     date_dict = dict()
#     for d in date_set:
#         date_dict[d] = pd.DataFrame()
#         date_index = origin_data[origin_data['actual_day'] == d].index
#         date_dict[d] = origin_data.loc[date_index]
#     return date_set, date_dict
#
#
# def __find_best_distribution_func():
#     # __data = [13, 56, 56, 7, 37, 7, 7, 7, 7, 11, 16, 7, 24, 14, 25, 10, 10, 10, 10, 10, 10, 10, 8, 23, 23, 14, 10, 41, 10, 10, 10, 8, 10, 10, 10, 10, 14, 14, 10, 14, 14, 10, 24, 10, 10, 15, 9, 13, 79, 13, 9, 14, 9, 9, 9, 9, 13, 10, 8, 8, 8, 8, 10, 8, 8, 8, 21, 11, 11, 13, 21, 21, 21, 21, 21, 21, 21, 8, 8, 8, 8, 51, 51, 21, 8, 8, 8, 8, 19, 8, 8, 14, 6, 37, 17, 15, 15, 17, 16, 37, 5, 9, 35, 35, 27, 5, 17, 4, 4, 6, 26, 1, 1, 1, 1, 1, 1, 1, 24, 23, 0, 0, 0, 9, 9, 9, 9, 9, 20, 20, 7, 8, 39, 7, 39, 33, 57, 39, 39, 39, 39, 40, 39, 15, 39, 20, 20, 20, 34, 22, 31, 0, 26, 25, 25, 25, 32, 32, 32, 32, 32, 32, 52, 185, 87, 1, 1, 15, 15, 15, 10, 15, 7, 7, 7, 7, 7, 10, 10, 7, 7, 7, 10, 10, 7, 7, 10, 18, 10, 20, 20, 20, 21, 20, 20, 9, 9, 6, 9, 17, 9, 9, 9, 9, 9, 6, 6, 6, 6, 6, 20, 20, 8, 16, 49, 20, 20, 20, 20, 20, 5, 7, 7, 11, 11, 15, 7, 11, 16, 16, 11, 7, 11, 9, 10, 10, 7, 10, 16, 14, 14, 7, 14, 14, 44, 7, 3, 5, 5, 6, 5, 44, 15, 43, 10, 10, 10, 10, 10, 44, 10, 9, 9, 9, 35, 10, 12, 13, 10, 13, 9, 9, 9, 9, 9, 12, 18, 9, 12, 12, 12, 9, 9, 12, 12, 8, 8, 8, 8, 8, 8, 8, 18, 8, 8, 8, 7, 7, 7, 7, 5, 5, 7, 9, 6, 7, 30, 7, 7, 32, 4, 3, 7, 3, 3, 37, 37, 4, 4, 4, 4, 4, 4, 37, 36, 4, 1, 17, 2, 2, 2, 2, 28, 35, 35, 35, 35, 35, 35, 35, 26, 35, 26, 35, 26, 26, 34, 26, 25, 25, 11, 25, 25, 25, 25, 25, 10, 23, 23, 8, 12, 12, 12, 8, 12, 8, 8, 8, 23, 23, 8, 8, 23, 9, 8, 1, 0, 8, 22, 25, 25, 7, 7, 22, 7, 22, 7, 7, 85, 20, 5, 32, 18, 26, 18, 68, 7, 51, 51, 7, 7, 16, 50, 7, 16, 53, 16, 7, 16, 15, 4, 12, 4, 22, 22, 4, 22, 22, 21, 22, 21, 12, 21, 5, 22, 22, 20, 14, 14, 14, 14, 8, 5, 8, 47, 8, 14, 16, 13, 13, 13, 16, 13, 7, 25, 4, 16, 7, 7, 7, 13, 7, 7, 16, 7, 13, 7, 16, 16, 13, 7, 21, 7, 21, 7, 7, 7, 7, 14, 7, 6, 12, 12, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 1, 19, 19, 20, 20, 1, 12, 12, 12, 12, 12, 12, 12, 15, 12, 9, 3, 7, 16, 15, 15, 15, 15, 3, 11, 11, 11, 8, 8, 11, 2, 16, 17, 8, 10, 10, 10, 10, 10, 7, 7, 10, 5, 7, 37, 37, 39, 39, 37, 37, 41, 39, 41, 8, 8, 8, 5, 9, 5, 5, 5, 30, 30, 37, 5, 30, 5, 30, 5, 30, 30, 5, 39, 10, 10, 10, 10, 5, 5, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8, 3, 12, 8, 2, 2, 2, 2, 7, 6, 1, 1, 1, 1, 30, 28, 28, 30, 28, 28, 30, 30, 28, 3, 3, 3, 3, 3, 27, 3, 2, 3, 3, 25, 25, 28, 28, 3, 9, 12, 12, 12, 27, 8, 8, 8, 12, 26, 1, 13, 8, 8, 8, 26, 13, 13, 13, 8, 13, 8, 8, 18, 16, 11, 15, 11, 7, 25, 15, 7, 25, 7, 19, 19, 19, 19, 19, 19, 27, 19, 14, 26, 26, 6, 51, 14, 3, 21, 11, 10, 10, 7, 10, 10, 7, 10, 18, 10, 18, 18, 10, 18, 18, 18, 10, 10, 10, 7, 7, 21, 20, 20, 20, 18, 18, 18, 9, 9, 9, 9, 6, 5, 8, 8, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 8, 8, 8, 7, 7, 17, 7, 7, 7, 7, 14, 14, 18, 14, 14, 14, 14, 14, 4, 14, 20, 14, 3, 11, 3, 11, 11, 3, 3, 11, 3, 13, 48, 48, 48, 48, 78, 78, 78, 78, 78, 10, 35, 16, 8, 8, 8, 8, 12, 8, 12, 2, 9, 1, 11, 9, 7, 12, 7, 7, 1, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 7, 11, 9, 11, 11, 11, 46, 46, 46, 46, 47, 8, 8, 8, 8, 6, 6, 8, 10, 11, 8, 6, 9, 7, 7, 7, 7, 30, 30, 7, 30, 30, 30, 7, 7, 7, 72, 7, 7, 7, 73, 30, 73, 4, 4, 4, 7, 4, 4, 4, 6, 6, 6, 3, 3, 2, 2, 23, 2, 5, 14, 5, 14, 5, 5, 21, 5, 14, 21, 14, 21, 5, 14, 21, 5, 19, 5, 5, 5, 4, 4, 4, 1, 1, 1, 1, 39, 68, 68, 68, 68, 68, 68, 68, 38, 38, 38, 68, 38, 38, 32, 33, 32, 0, 32, 32, 34, 32, 0, 0, 0, 0, 0, 0, 1, 3, 4, 2, 10, 1, 1, 1, 1, 10, 17, 17, 1, 1, 36, 23, 10, 15, 4, 1, 9, 9, 9, 9, 9, 16, 14, 9, 18, 16, 22, 16, 31, 16, 16, 31, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24, 15, 13, 61, 14, 18, 18, 33, 25, 25, 29, 29, 30, 30, 59, 29, 29, 29, 29, 29, 31, 29, 28, 28, 8, 8, 8, 8, 10, 8, 8, 16, 8, 1, 8, 8, 8, 8, 8, 16, 16, 10, 10, 10, 6, 10, 10, 14, 18, 18, 26, 7, 21, 5, 53, 53, 53, 74, 74, 74, 74, 6, 8, 54, 52, 52, 52, 52, 19, 19, 19, 19, 19, 19, 21, 19, 20, 24, 24, 24, 23, 24, 23, 24, 24, 24, 20, 18, 14, 13, 5, 13, 23, 24, 24, 23, 23, 24, 23, 45, 24, 51, 7, 7, 7, 18, 7, 18, 13, 13, 7, 7, 7, 18, 18, 7, 18, 18, 23, 45, 24, 23, 12, 7, 12, 12, 13, 12, 16, 12, 13, 16, 16, 17, 16, 16, 16, 16, 16, 5, 16, 5, 11, 11, 11, 11, 11, 11, 11, 10, 15, 10, 10, 8, 10, 8, 10, 10, 20, 10, 9, 16, 16, 15, 16, 15, 19, 19, 19, 19, 20, 20, 20, 20, 19, 19, 19, 19, 19, 20, 19, 8, 9, 9, 9, 9, 8, 13, 8, 8, 23, 13, 13, 13, 12, 13, 12, 13, 13, 12, 12, 7, 7, 11, 0, 10, 43, 43, 43, 43, 5, 5, 5, 5, 4, 9, 9, 11, 4, 13, 13, 8, 40, 9, 9, 3, 9, 15, 33, 54, 54, 40, 15, 12, 15, 40, 9, 12, 9, 9, 8, 7, 7, 7, 7, 7, 10, 7, 5, 7, 7, 7, 7, 6, 6, 11, 7, 5, 6, 6, 69, 40, 67, 62, 4, 39, 4, 6, 5, 5, 5, 6, 6, 5, 9, 9, 9, 9, 9, 6, 5, 5, 5, 5, 5, 4, 4, 4, 7, 4, 35, 3, 7, 3, 7, 3, 7, 7, 3, 8, 4, 3, 6, 3, 36, 36, 2, 6, 2, 6, 2, 2, 2, 2, 5, 5, 5, 5, 53, 53, 53, 53, 35, 35, 3, 31, 33, 33, 33, 32, 32, 2, 3, 2, 0, 0, 23, 34, 51, 29, 51, 27, 30, 27, 21, 7, 7, 36, 36, 8, 6, 31, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 9, 20, 6, 8, 8, 8, 8, 6, 29, 5, 5, 5, 5, 5, 7, 15, 5, 5, 7, 15, 5, 4, 4, 4, 4, 4, 6, 3, 3, 3, 3, 13, 34, 34, 34, 35, 2, 2, 34, 2, 33, 2, 2, 0, 0, 0, 0, 2, 13, 1, 13, 1, 1, 9, 13, 29, 13, 0, 0, 27, 27, 23, 23, 8, 11, 7, 11, 16, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 19, 19, 19, 19, 25, 5, 5, 17, 16, 17, 13, 5, 5, 15, 12, 12, 12, 21, 12, 21, 15, 9, 15, 9, 9, 9, 9, 15, 12, 9, 9, 12, 19, 19, 4, 8, 14, 19, 18, 15, 9, 7, 4, 7, 13, 13, 13, 19, 18, 13, 13, 13, 13, 14, 10, 15, 15, 12, 12, 11, 11, 11, 11, 11, 9, 7, 7, 15, 15, 15, 2, 13, 13, 14, 13, 13, 14, 13, 13, 14, 8, 13, 8, 13, 13, 14, 8, 10, 11, 6, 6, 12, 12, 5, 5, 5, 5, 11, 11, 10, 19, 10, 4, 4, 7, 4, 4, 4, 9, 8, 9, 9, 6, 6, 2, 2, 8, 7, 7, 14, 39, 39, 7, 29, 3, 21, 3, 31, 21, 1, 1, 1, 15, 25, 12, 12, 14, 12, 19, 12, 12, 33, 12, 12, 19, 12, 12, 12, 12, 12, 13, 10, 17, 10, 17, 31, 31, 31, 31, 31, 29, 16, 11, 7, 13, 13, 13, 25, 16, 5, 16, 23, 37, 5, 16, 5, 15, 15, 16, 22, 16, 16, 40, 25, 16, 16, 7, 7, 25, 25, 25, 15, 12, 13, 16, 15, 15, 16, 11, 11, 6, 17, 19, 6, 9, 9, 9, 9, 9, 9, 9, 9, 6, 9, 15, 33, 19, 19, 8, 19, 19, 9, 8, 8, 9, 5, 6, 6, 17, 6, 6, 15, 17, 12, 15, 13, 27, 12, 15, 15, 5, 5, 13, 6, 6, 15, 6, 12, 12, 12, 13, 12, 12, 10, 10, 10, 24, 13, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 9, 9, 6, 6, 6, 7, 9, 10, 10, 10, 5, 5, 5, 5, 21, 2, 21, 21, 6, 6, 6, 18, 4, 21, 21, 21, 21, 6, 21, 21, 21, 5, 3, 0, 2, 14, 15, 15, 12, 14, 14, 11, 4]
#     __data = [15, 9, 13, 13, 9, 9, 9, 9, 9, 13, 13, 23, 9, 9, 9, 1, 1, 9, 9, 9, 9, 9, 9, 4, 28, 8, 12, 7, 51, 51, 12, 7, 7, 16, 50, 7, 16, 53, 16, 5, 23, 24, 9, 9, 9, 9, 12, 8, 12, 12, 7, 12, 12, 12, 16, 15, 22, 9, 6, 10, 10, 10, 10, 5, 5, 10, 1, 1, 9, 9, 9, 9, 6, 24, 24, 24, 23, 24, 23, 24, 24, 5, 24, 5, 5, 5, 20, 18, 14, 13, 5, 13, 23, 24, 24, 23, 23, 24, 23, 45, 24, 51, 7, 7, 7, 18, 7, 18, 13, 13, 13, 7, 7, 7, 18, 18, 7, 18, 18, 23, 45, 24, 23, 13, 7, 13, 13, 23, 23, 13, 6, 9, 8, 9, 14, 14, 14, 14, 7, 7, 51, 51, 11, 10]
#     f = Fitter(__data)
#     f.fit()
#     f.summary()
#     # f.hist()
#     plt.show()
#
#
# def fit_distribution_func(data_set: set, data_dict: dict):
#     for s in data_set:
#         lead_time_data = list(data_dict[s]['lead_time'])
#         # lead_time_data = lead_time_data.values
#         print(lead_time_data)
#
#
# if __name__ == "__main__":
#     wgbez_set, wgbez_dict = data_split_by_date()
#     fit_distribution_func(wgbez_set, wgbez_dict)
#     __find_best_distribution_func()
