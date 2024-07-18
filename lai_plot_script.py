import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shutil

fontsize = 20
nice_fonts = {
    "font.family": "serif",
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
}
matplotlib.rcParams.update(nice_fonts)

import os
from utils import load, make_dir
import numpy as np
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--phys', type=str, default='tsp', help='The physics system to be studied')
parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate for gd')
parser.add_argument('--dim', type=int, default=20, help='The dimensionality of the data')
parser.add_argument('--sigma', type=float, default=0.0, help='The standard deviation of the Gaussian noise')
parser.add_argument('--x_lim', type=int, default=-1, help='The x-axis limit for the plots')
args = parser.parse_args()

folder_path = f'exp/{args.phys}/lr_{args.lr}/dim_{args.dim}/sigma_{args.sigma}'
plot_path = f'plots/{args.phys}/lr_{args.lr}/dim_{args.dim}/sigma_{args.sigma}'

if os.path.exists(plot_path):
    shutil.rmtree(plot_path)
    print(f"Removed existing directory: {plot_path}")
make_dir(plot_path)

num_exp = len(glob(f'{folder_path}/exp_*'))
experiments = [f'exp_{i}' for i in range(num_exp)]

def process_data(dataset, data_type):
    function_values = np.array(dataset[f'function_values_{data_type}'])
    eigen_values = np.array([np.max(np.abs(x)) for x in dataset[f"eigen_values_{data_type}"]])
    lipshitz_diagonal_values = np.array([np.mean(np.abs(x)) for x in dataset[f"lip_diag_values_{data_type}"]])
    max_lipshitz_diagonal_values = np.array([np.max(np.abs(x)) for x in dataset[f"lip_diag_values_{data_type}"]])
    mean_eigen_lipshitz_ratio = eigen_values / lipshitz_diagonal_values
    max_eigen_lipshitz_ratio = eigen_values / max_lipshitz_diagonal_values
    return function_values, mean_eigen_lipshitz_ratio, max_eigen_lipshitz_ratio

gd_function_values_list = []
rcd_function_values_list = []
bcd_function_values_list = []
bcd_c_function_values_list = []
bcd_g_function_values_list = []
bcd_reg_function_values_list = []

gd_mean_eigen_lipshitz_ratios = []
rcd_mean_eigen_lipshitz_ratios = []
bcd_mean_eigen_lipshitz_ratios = []
bcd_c_mean_eigen_lipshitz_ratios = []
bcd_g_mean_eigen_lipshitz_ratios = []
bcd_reg_mean_eigen_lipshitz_ratios = []

gd_max_eigen_lipshitz_ratios = []
rcd_max_eigen_lipshitz_ratios = []
bcd_max_eigen_lipshitz_ratios = []
bcd_c_max_eigen_lipshitz_ratios = []
bcd_g_max_eigen_lipshitz_ratios = []
bcd_reg_max_eigen_lipshitz_ratios = []

for experiment in experiments:
    file_path = os.path.join(folder_path, experiment, 'data_dict.pkl')
    data = load(file_path)
    

    for key in data.keys():
        data[key] = np.array(data[key])

    gd_function_values, gd_mean_eigen_lipshitz_ratio, gd_max_eigen_lipshitz_ratio = process_data(data, 'gd')
    rcd_function_values, rcd_mean_eigen_lipshitz_ratio, rcd_max_eigen_lipshitz_ratio = process_data(data, 'rcd')
    bcd_function_values, bcd_mean_eigen_lipshitz_ratio, bcd_max_eigen_lipshitz_ratio = process_data(data, 'bcd')
    bcd_c_function_values, bcd_c_mean_eigen_lipshitz_ratio, bcd_c_max_eigen_lipshitz_ratio = process_data(data, 'bcd_c')
    bcd_g_function_values, bcd_g_mean_eigen_lipshitz_ratio, bcd_g_max_eigen_lipshitz_ratio = process_data(data, 'bcd_g')
    bcd_reg_function_values, bcd_reg_mean_eigen_lipshitz_ratio, bcd_reg_max_eigen_lipshitz_ratio = process_data(data, 'bcd_reg')
    
    gd_function_values_list.append(gd_function_values)
    rcd_function_values_list.append(rcd_function_values)
    bcd_function_values_list.append(bcd_function_values)
    bcd_c_function_values_list.append(bcd_c_function_values)
    bcd_g_function_values_list.append(bcd_g_function_values)
    bcd_reg_function_values_list.append(bcd_reg_function_values)
    
    gd_mean_eigen_lipshitz_ratios.append(gd_mean_eigen_lipshitz_ratio)
    rcd_mean_eigen_lipshitz_ratios.append(rcd_mean_eigen_lipshitz_ratio)
    bcd_mean_eigen_lipshitz_ratios.append(bcd_mean_eigen_lipshitz_ratio)
    bcd_c_mean_eigen_lipshitz_ratios.append(bcd_c_mean_eigen_lipshitz_ratio)
    bcd_g_mean_eigen_lipshitz_ratios.append(bcd_g_mean_eigen_lipshitz_ratio)
    bcd_reg_mean_eigen_lipshitz_ratios.append(bcd_reg_mean_eigen_lipshitz_ratio)
    
    gd_max_eigen_lipshitz_ratios.append(gd_max_eigen_lipshitz_ratio)
    rcd_max_eigen_lipshitz_ratios.append(rcd_max_eigen_lipshitz_ratio)
    bcd_max_eigen_lipshitz_ratios.append(bcd_max_eigen_lipshitz_ratio)
    bcd_c_max_eigen_lipshitz_ratios.append(bcd_c_max_eigen_lipshitz_ratio)
    bcd_g_max_eigen_lipshitz_ratios.append(bcd_g_max_eigen_lipshitz_ratio)
    bcd_reg_max_eigen_lipshitz_ratios.append(bcd_reg_max_eigen_lipshitz_ratio)

gd_function_values_df = pd.DataFrame(gd_function_values_list)
rcd_function_values_df = pd.DataFrame(rcd_function_values_list)
bcd_function_values_df = pd.DataFrame(bcd_function_values_list)
bcd_c_function_values_df = pd.DataFrame(bcd_c_function_values_list)
bcd_g_function_values_df = pd.DataFrame(bcd_g_function_values_list)
bcd_reg_function_values_df = pd.DataFrame(bcd_reg_function_values_list)

gd_mean_ratios_df = pd.DataFrame(gd_mean_eigen_lipshitz_ratios)
rcd_mean_ratios_df = pd.DataFrame(rcd_mean_eigen_lipshitz_ratios)
bcd_mean_ratios_df = pd.DataFrame(bcd_mean_eigen_lipshitz_ratios)
bcd_c_mean_ratios_df = pd.DataFrame(bcd_c_mean_eigen_lipshitz_ratios)
bcd_g_mean_ratios_df = pd.DataFrame(bcd_g_mean_eigen_lipshitz_ratios)
bcd_reg_mean_ratios_df = pd.DataFrame(bcd_reg_mean_eigen_lipshitz_ratios)

gd_max_ratios_df = pd.DataFrame(gd_max_eigen_lipshitz_ratios)
rcd_max_ratios_df = pd.DataFrame(rcd_max_eigen_lipshitz_ratios)
bcd_max_ratios_df = pd.DataFrame(bcd_max_eigen_lipshitz_ratios)
bcd_c_max_ratios_df = pd.DataFrame(bcd_c_max_eigen_lipshitz_ratios)
bcd_g_max_ratios_df = pd.DataFrame(bcd_g_max_eigen_lipshitz_ratios)
bcd_reg_max_ratios_df = pd.DataFrame(bcd_reg_max_eigen_lipshitz_ratios)

mean_gd_values = np.array(gd_function_values_df.mean(), dtype=float)
std_gd_values = np.array(gd_function_values_df.std(), dtype=float)
mean_rcd_values = np.array(rcd_function_values_df.mean(), dtype=float)
std_rcd_values = np.array(rcd_function_values_df.std(), dtype=float)
mean_bcd_values = np.array(bcd_function_values_df.mean(), dtype=float)
std_bcd_values = np.array(bcd_function_values_df.std(), dtype=float)
mean_bcd_c_values = np.array(bcd_c_function_values_df.mean(), dtype=float)
std_bcd_c_values = np.array(bcd_c_function_values_df.std(), dtype=float)
mean_bcd_g_values = np.array(bcd_g_function_values_df.mean(), dtype=float)
std_bcd_g_values = np.array(bcd_g_function_values_df.std(), dtype=float)
mean_bcd_reg_values = np.array(bcd_reg_function_values_df.mean(), dtype=float)
std_bcd_reg_values = np.array(bcd_reg_function_values_df.std(), dtype=float)

mean_gd_ratios = np.array(gd_mean_ratios_df.mean(), dtype=float)
std_gd_ratios = np.array(gd_mean_ratios_df.std(), dtype=float)
mean_rcd_ratios = np.array(rcd_mean_ratios_df.mean(), dtype=float)
std_rcd_ratios = np.array(rcd_mean_ratios_df.std(), dtype=float)
mean_bcd_ratios = np.array(bcd_mean_ratios_df.mean(), dtype=float)
std_bcd_ratios = np.array(bcd_mean_ratios_df.std(), dtype=float)
mean_bcd_c_ratios = np.array(bcd_c_mean_ratios_df.mean(), dtype=float)
std_bcd_c_ratios = np.array(bcd_c_mean_ratios_df.std(), dtype=float)
mean_bcd_g_ratios = np.array(bcd_g_mean_ratios_df.mean(), dtype=float)
std_bcd_g_ratios = np.array(bcd_g_mean_ratios_df.std(), dtype=float)
mean_bcd_reg_ratios = np.array(bcd_reg_mean_ratios_df.mean(), dtype=float)
std_bcd_reg_ratios = np.array(bcd_reg_mean_ratios_df.std(), dtype=float)

mean_gd_max_ratios = np.array(gd_max_ratios_df.mean(), dtype=float)
std_gd_max_ratios = np.array(gd_max_ratios_df.std(), dtype=float)
mean_rcd_max_ratios = np.array(rcd_max_ratios_df.mean(), dtype=float)
std_rcd_max_ratios = np.array(rcd_max_ratios_df.std(), dtype=float)
mean_bcd_max_ratios = np.array(bcd_max_ratios_df.mean(), dtype=float)
std_bcd_max_ratios = np.array(bcd_max_ratios_df.std(), dtype=float)
mean_bcd_c_max_ratios = np.array(bcd_c_max_ratios_df.mean(), dtype=float)
std_bcd_c_max_ratios = np.array(bcd_c_max_ratios_df.std(), dtype=float)
mean_bcd_g_max_ratios = np.array(bcd_g_max_ratios_df.mean(), dtype=float)
std_bcd_g_max_ratios = np.array(bcd_g_max_ratios_df.std(), dtype=float)
mean_bcd_reg_max_ratios = np.array(bcd_reg_max_ratios_df.mean(), dtype=float)
std_bcd_reg_max_ratios = np.array(bcd_reg_max_ratios_df.std(), dtype=float)

plt.figure()
plt.plot(mean_gd_values, color='r', label='GD')
plt.plot(mean_rcd_values, color='b', label='RCD')
plt.plot(mean_bcd_values, color='k', label='BCD')
plt.plot(mean_bcd_c_values, color='g', label='BCD_C')
plt.plot(mean_bcd_g_values, color='m', label='BCD_G')
plt.plot(mean_bcd_reg_values, color='c', label='BCD_REG')
plt.fill_between(range(len(mean_gd_values)), 
                 mean_gd_values - std_gd_values, 
                 mean_gd_values + std_gd_values, 
                 color='r', alpha=0.2)
plt.fill_between(range(len(mean_rcd_values)), 
                 mean_rcd_values - std_rcd_values, 
                 mean_rcd_values + std_rcd_values, 
                 color='b', alpha=0.2)
plt.fill_between(range(len(mean_bcd_values)), 
                 mean_bcd_values - std_bcd_values, 
                 mean_bcd_values + std_bcd_values, 
                 color='k', alpha=0.2)
plt.fill_between(range(len(mean_bcd_c_values)), 
                 mean_bcd_c_values - std_bcd_c_values, 
                 mean_bcd_c_values + std_bcd_c_values, 
                 color='g', alpha=0.2)
plt.fill_between(range(len(mean_bcd_g_values)), 
                 mean_bcd_g_values - std_bcd_g_values, 
                 mean_bcd_g_values + std_bcd_g_values, 
                 color='m', alpha=0.2)
plt.fill_between(range(len(mean_bcd_reg_values)), 
                 mean_bcd_reg_values - std_bcd_reg_values, 
                 mean_bcd_reg_values + std_bcd_reg_values, 
                 color='c', alpha=0.2)
plt.title('Energy Ratio')
plt.xlabel('Iterations')
plt.ylabel('Energy ratio: $E / E_{GS}$')
plt.legend()
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/energy_HM.png')
plt.clf()

plt.figure()
times_gd = np.arange(len(mean_gd_values)) * args.dim * 2
plt.plot(times_gd, mean_gd_values, color='r', label='GD')
times_rcd = np.arange(len(mean_rcd_values)) * 2
plt.plot(times_rcd, mean_rcd_values, color='b', label='RCD')
times_bcd = np.arange(len(mean_bcd_values)) * 3
plt.plot(times_bcd, mean_bcd_values, color='k', label='BCD')
times_bcd_c = np.arange(len(mean_bcd_c_values)) * 3
plt.plot(times_bcd_c, mean_bcd_c_values, color='g', label='BCD_C')
times_bcd_g = np.arange(len(mean_bcd_g_values)) * 3
plt.plot(times_bcd_g, mean_bcd_g_values, color='m', label='BCD_G')
times_bcd_reg = np.arange(len(mean_bcd_reg_values)) * data['fevl_num_each_iter_reg']
plt.plot(times_bcd_reg, mean_bcd_reg_values, color='c', label='BCD_REG')
plt.fill_between(times_gd, 
                 mean_gd_values - std_gd_values, 
                 mean_gd_values + std_gd_values, 
                 color='r', alpha=0.2)
plt.fill_between(times_rcd, 
                 mean_rcd_values - std_rcd_values, 
                 mean_rcd_values + std_rcd_values, 
                 color='b', alpha=0.2)
plt.fill_between(times_bcd, 
                 mean_bcd_values - std_bcd_values, 
                 mean_bcd_values + std_bcd_values, 
                 color='k', alpha=0.2)
plt.fill_between(times_bcd_c, 
                 mean_bcd_c_values - std_bcd_c_values, 
                 mean_bcd_c_values + std_bcd_c_values, 
                 color='g', alpha=0.2)
plt.fill_between(times_bcd_g, 
                 mean_bcd_g_values - std_bcd_g_values, 
                 mean_bcd_g_values + std_bcd_g_values, 
                 color='m', alpha=0.2)
plt.fill_between(times_bcd_reg, 
                 mean_bcd_reg_values - std_bcd_reg_values, 
                 mean_bcd_reg_values + std_bcd_reg_values, 
                 color='c', alpha=0.2)
plt.title('Energy Ratio')
plt.xlabel('Number of function evaluations')
plt.ylabel('Energy ratio: $ E / E_{GS}$')
plt.legend()
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
plt.xlim(0, min(max(times_rcd), max(times_bcd), max(times_bcd_c), max(times_bcd_g), max(times_gd), max(times_bcd_reg)))
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/energy_HM_fun_evals.png')
plt.clf()

plt.figure()
plt.plot(mean_gd_ratios, color='r', label='GD')
plt.plot(mean_rcd_ratios, color='b', label='RCD')
plt.plot(mean_bcd_ratios, color='k', label='BCD')
plt.plot(mean_bcd_c_ratios, color='g', label='BCD_C')
plt.plot(mean_bcd_g_ratios, color='m', label='BCD_G')
plt.plot(mean_bcd_reg_ratios, color='c', label='BCD_REG')
plt.fill_between(range(len(mean_gd_ratios)), 
                 mean_gd_ratios - std_gd_ratios, 
                 mean_gd_ratios + std_gd_ratios, 
                 color='r', alpha=0.2)
plt.fill_between(range(len(mean_rcd_ratios)), 
                 mean_rcd_ratios - std_rcd_ratios, 
                 mean_rcd_ratios + std_rcd_ratios, 
                 color='b', alpha=0.2)
plt.fill_between(range(len(mean_bcd_ratios)), 
                 mean_bcd_ratios - std_bcd_ratios, 
                 mean_bcd_ratios + std_bcd_ratios, 
                 color='k', alpha=0.2)
plt.fill_between(range(len(mean_bcd_c_ratios)), 
                 mean_bcd_c_ratios - std_bcd_c_ratios, 
                 mean_bcd_c_ratios + std_bcd_c_ratios, 
                 color='g', alpha=0.2)
plt.fill_between(range(len(mean_bcd_g_ratios)), 
                 mean_bcd_g_ratios - std_bcd_g_ratios, 
                 mean_bcd_g_ratios + std_bcd_g_ratios, 
                 color='m', alpha=0.2)
plt.fill_between(range(len(mean_bcd_reg_ratios)), 
                 mean_bcd_reg_ratios - std_bcd_reg_ratios, 
                 mean_bcd_reg_ratios + std_bcd_reg_ratios, 
                 color='c', alpha=0.2)
plt.title('Lipschitz Constant Ratio')
plt.xlabel('Steps')
plt.ylabel('L / $L_{avg}$')
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/lip_HM.png')
plt.clf()

plt.figure()
plt.plot(mean_gd_max_ratios, color='r', label='GD')
plt.plot(mean_rcd_max_ratios, color='b', label='RCD')
plt.plot(mean_bcd_c_max_ratios, color='g', label='BCD_C')
plt.plot(mean_bcd_g_max_ratios, color='m', label='BCD_G')
plt.plot(mean_bcd_reg_max_ratios, color='c', label='BCD_REG')
plt.fill_between(range(len(mean_gd_max_ratios)), 
                 mean_gd_max_ratios - std_gd_max_ratios, 
                 mean_gd_max_ratios + std_gd_max_ratios, 
                 color='r', alpha=0.2)
plt.fill_between(range(len(mean_rcd_max_ratios)), 
                 mean_rcd_max_ratios - std_rcd_max_ratios, 
                 mean_rcd_max_ratios + std_rcd_max_ratios, 
                 color='b', alpha=0.2)
plt.fill_between(range(len(mean_bcd_c_max_ratios)), 
                 mean_bcd_c_max_ratios - std_bcd_c_max_ratios, 
                 mean_bcd_c_max_ratios + std_bcd_c_max_ratios, 
                 color='g', alpha=0.2)
plt.fill_between(range(len(mean_bcd_g_max_ratios)), 
                 mean_bcd_g_max_ratios - std_bcd_g_max_ratios, 
                 mean_bcd_g_max_ratios + std_bcd_g_max_ratios, 
                 color='m', alpha=0.2)
plt.fill_between(range(len(mean_bcd_reg_max_ratios)), 
                 mean_bcd_reg_max_ratios - std_bcd_reg_max_ratios, 
                 mean_bcd_reg_max_ratios + std_bcd_reg_max_ratios, 
                 color='c', alpha=0.2)
plt.title('Lipschitz Constant Ratio')
plt.xlabel('Steps')
plt.ylabel('L / $L_{max}$')
if args.x_lim > 0: 
    plt.xlim(0, args.x_lim)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'{plot_path}/lip_max_HM.png')
plt.clf()
