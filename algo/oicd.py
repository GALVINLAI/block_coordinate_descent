import jax
import jax.random as jrd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import minimize

'''
The oicd algorithm here is compatible with Ding's code
'''

def construct_Es_inv(s, Omegas):
    """
    构造块对角矩阵 E_s^{-1}，每个块为 B_i^T，其中 B_i 为旋转矩阵。
    
    参数:
    s -- 标量，旋转角度的尺度。
    Omegas -- 列表，包含 Omega_i 的值。
    
    返回:
    E_s_inv -- 构造的块对角矩阵 E_s^{-1}。
    """
    # 计算旋转矩阵 B_i^T
    num_blocks = len(Omegas) + 1  # 第一块是 1x1 矩阵 [1]，之后每个块为 2x2 矩阵
    total_size = num_blocks * 2 - 1  # 计算总大小
    
    # 初始化一个全零矩阵，大小为 total_size x total_size
    E_s_inv = np.zeros((total_size, total_size))
    
    # 设置第一块 1x1 矩阵
    E_s_inv[0, 0] = 1
    
    # 填充后续的旋转矩阵块 B_i^T
    for i, Omega_i in enumerate(Omegas):
        # 构造每个 B_i
        B_i = np.array([[np.cos(Omega_i * s), np.sin(Omega_i * s)],
                        [-np.sin(Omega_i * s), np.cos(Omega_i * s)]])
        # 将 B_i^T 填入矩阵 E_s_inv
        E_s_inv[2*i+1:2*i+3, 2*i+1:2*i+3] = B_i.T
    
    return E_s_inv


def oicd(f, generators_dict, initial_point, num_iterations, sigma, key,
                             problem_name, 
                             opt_goal='max', 
                             plot_subproblem=False,
                             cyclic_mode=False, 
                             subproblem_iter=20,
                             fevl_num_each_iter=6,
                             mode='general',
                             alpha=0.8):
    
    theta = initial_point
    best_point = theta
    best_value = f(theta)
    function_values = [best_value]
    hat_f_value = best_value
    hat_function_values = [best_value]

    # if plot_subproblem:
    #     output_dir = setup_plot_subproblem(problem_name)

    print("-"*100)
    
    t = trange(num_iterations, desc="Bar desc", leave=True)
    m = len(theta)
    for i in t:
        if cyclic_mode:
            j = i % m
        else:
            key, subkey = jrd.split(key)
            j = jrd.randint(subkey, shape=(), minval=0, maxval=m)

        # theta_old = theta.copy()

        interp_points = generators_dict[f'Generator_{j}']["opt_interp_points"]
        inv_A = generators_dict[f'Generator_{j}']['inverse_interp_matrix']
        Omegas = generators_dict[f'Generator_{j}']['omega_set']

        shift = theta[j] - interp_points[0]
        interp_points += shift
        E_s_inv = construct_Es_inv(shift, Omegas)
        
        fun_vals = [hat_f_value]
        for point in interp_points[1:]:
            key, subkey = jrd.split(key)
            # theta_old[j] = point
            fun_val = f(theta.at[j].set(point)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        fun_vals = np.array(fun_vals)

        hat_z = E_s_inv @ (inv_A @ fun_vals)

        # 定义 f(x) 的计算，包括 t(x) 的计算
        def hat_f(x): # appr_single_var_fun
            # 计算 t_x 时，直接将 1 / np.sqrt(2) 加到每个值中
            r= len(Omegas)
            t_x = np.array([1 / np.sqrt(2)] + [func(Omegas[k] * x).item() for k in range(r) for func in (np.cos, np.sin)])
            return np.dot(t_x, hat_z)

        # 使用 scipy.optimize.minimize 最小化目标函数
        initial_guess = theta[j]  # 初始值一定要选择当前值

        # 设置最大迭代步数
        options = {'maxiter': subproblem_iter, 'disp': False}

        if opt_goal == 'max':
            HAT_f = lambda x: - hat_f(x)
        else:
            HAT_f = hat_f

        # 运行优化算法求解子问题
        result = minimize(HAT_f, initial_guess, options=options)

        # 获取最优解和最小值
        theta_star = result.x.item()  # 最优解
        hat_f_value = result.fun  # 最小值

        # 如果是最大化问题，取负值
        if opt_goal == 'max':
            hat_f_value = -hat_f_value

        theta = theta.at[j].set(theta_star)
        next_point = theta
        next_value = f(theta)
        function_values.append(next_value)
        hat_function_values.append(hat_f_value)

        if next_value > best_value:
            best_point = next_point
            best_value = next_value

        message = f"Iteration: {i}, Value: {next_value}, Coord j: {j}({m})"
        t.set_description(f"[OICD-{mode}] Processing %s" % message)
        t.refresh()

    print(f"[OICD-{mode}] Final value of f:", best_value)

    return best_point, best_value, function_values






