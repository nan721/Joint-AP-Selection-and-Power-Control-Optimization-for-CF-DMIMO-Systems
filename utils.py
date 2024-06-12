# 负责画图及创建指定文件夹。本文件不依赖于任何文件

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(figure_file, 'exam.png'))
    plt.show()


def plot_compare_curve(index, RL_values, NN_values, title, ylabel, figure_file):
    plt.figure()
    plt.plot(index, RL_values, linestyle='-', color='r', label='RL')
    plt.plot(index, NN_values, linestyle='-', color='b', label="NN")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('Position Index')
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(figure_file, 'compare_curve.png'))
    plt.show()


def plot_triple_compare_curve(index, RL_values, NN_values, GM_values, title, ylabel, figure_file):
    plt.figure()
    plt.plot(index, RL_values, linestyle='-', color='r', label='RL')
    plt.plot(index, NN_values, linestyle='-', color='b', label="NN")
    plt.plot(index, GM_values, linestyle='-', color='g', label="GM")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('Position Index')
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(figure_file, 'compare_curve.png'))
    plt.show()


def plot_bar(NN_values, GM_values, PSO_c_values, PSO_d_values, RL_values, x_label, title, figure_file):
    plt.figure()
    x = list(range(len(NN_values)))
    total_width, n = 0.5, 5
    width = total_width / n
    plt.bar(x, NN_values, width=width, label='NN', fc='b', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, PSO_d_values, width=width, label='IBPSO-D', fc='m', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, RL_values, width=width, label='RL', fc='r', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, PSO_c_values, width=width, label='IBPSO-C', fc='y', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, GM_values, width=width, label='GM', fc='g', tick_label='')
    plt.xticks([])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel('Sum Rate (Mbit/s)')
    plt.title('')
    plt.savefig(os.path.join(figure_file, title + '.png'))
    plt.show()


def plot_CDF(NN_values, GM_values, PSO_c_values, PSO_d_values, RL_values, title, figure_file):
    min_range = min(min(NN_values), min(GM_values), min(PSO_c_values), min(PSO_d_values), min(RL_values)) - 0.5
    max_range = max(max(NN_values), max(GM_values), max(PSO_c_values), max(PSO_d_values), max(RL_values)) + 0.5
    NN_hist, bin_edges = np.histogram(NN_values, range=(min_range, max_range), bins=1000)
    GM_hist, bin_edges = np.histogram(GM_values, range=(min_range, max_range), bins=1000)
    PSO_c_hist, bin_edges = np.histogram(PSO_c_values, range=(min_range, max_range), bins=1000)
    PSO_d_hist, bin_edges = np.histogram(PSO_d_values, range=(min_range, max_range), bins=1000)
    RL_hist, bin_edges = np.histogram(RL_values, range=(min_range, max_range), bins=1000)
    NN_cdf = np.cumsum(NN_hist / sum(NN_hist))
    GM_cdf = np.cumsum(GM_hist / sum(GM_hist))
    PSO_c_cdf = np.cumsum(PSO_c_hist / sum(PSO_c_hist))
    PSO_d_cdf = np.cumsum(PSO_d_hist / sum(PSO_d_hist))
    RL_cdf = np.cumsum(RL_hist / sum(RL_hist))
    plt.plot(bin_edges[1:], NN_cdf, label='NN', color='b')
    plt.plot(bin_edges[1:], GM_cdf, label='GM', color='g')
    plt.plot(bin_edges[1:], PSO_c_cdf, label='IBPSO-C', color='y')
    plt.plot(bin_edges[1:], PSO_d_cdf, label='IBPSO-D', color='m')
    plt.plot(bin_edges[1:], RL_cdf, label='RL', color='r')
    plt.legend(loc='best')
    plt.title('')
    plt.xlabel('Sum Rate (Mbit/s)')
    plt.ylabel('CDF')
    plt.grid()
    plt.savefig(os.path.join(figure_file, title + '.png'))
    plt.show()


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')


def plot_bar_withMEN(NN_values, GM_values, MEN_values, RL_values, x_label, title, figure_file):
    plt.figure()
    x = list(range(len(NN_values)))
    total_width, n = 0.5, 4
    width = total_width / n
    plt.bar(x, NN_values, width=width, label='NN', fc='b', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, MEN_values, width=width, label='MEN', fc='m', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, RL_values, width=width, label='RL', fc='r', tick_label='')
    for i in range(len(x)):
        x[i] += width
    plt.bar(x, GM_values, width=width, label='GM', fc='g', tick_label='')
    plt.xticks([])
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel('Sum Rate (Mbit/s)')
    plt.title('')
    plt.savefig(os.path.join(figure_file, title + '.png'))
    plt.show()


def plot_CDF_withMEN(NN_values, GM_values, MEN_values, RL_values, title, figure_file):
    min_range = min(min(NN_values), min(MEN_values), min(RL_values)) - 0.5
    max_range = max(max(NN_values), max(MEN_values), max(RL_values)) + 0.5
    max_range = 600
    NN_hist, bin_edges = np.histogram(NN_values, range=(min_range, max_range), bins=1000)
    GM_hist, bin_edges = np.histogram(GM_values, range=(min_range, max_range), bins=1000)
    MEN_hist, bin_edges = np.histogram(MEN_values, range=(min_range, max_range), bins=1000)
    RL_hist, bin_edges = np.histogram(RL_values, range=(min_range, max_range), bins=1000)
    NN_cdf = np.cumsum(NN_hist / sum(NN_hist))
    GM_cdf = np.cumsum(GM_hist / sum(GM_hist))
    MEN_cdf = np.cumsum(MEN_hist / sum(MEN_hist))
    RL_cdf = np.cumsum(RL_hist / sum(RL_hist))
    plt.plot(bin_edges[1:], NN_cdf, label='NN', color='b')
    # plt.plot(bin_edges[1:], GM_cdf, label='GM', color='g')
    plt.plot(bin_edges[1:], MEN_cdf, label='MEN', color='m')
    plt.plot(bin_edges[1:], RL_cdf, label='RL', color='r')
    plt.legend(loc='best')
    plt.title('')
    plt.xlabel('Sum Rate (Mbit/s)')
    plt.ylabel('CDF')
    plt.grid()
    plt.savefig(os.path.join(figure_file, title + '.png'))
    plt.show()

def plot_CDF_4_14( SAC8_values,title, figure_file):
    min_range = min( SAC8_values) - 0.5
    max_range = max( SAC8_values) + 0.5
    # max_range = 600

    SAC8_hist, bin_edges = np.histogram(SAC8_values, range=(min_range, max_range), bins=1000)


    SAC8_cdf = np.cumsum(SAC8_hist / sum(SAC8_hist))



    plt.plot(bin_edges[1:], SAC8_cdf, label='SACu0', color='r')
    plt.legend(loc='best')
    plt.title('')
    plt.xlabel('Sum Rate (Mbit/s)')
    plt.ylabel('CDF')
    plt.grid()
    plt.savefig(os.path.join(figure_file, title + '.png'))
    plt.show()