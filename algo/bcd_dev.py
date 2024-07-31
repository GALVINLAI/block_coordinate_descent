import jax
import jax.random as jrd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import trange

'''
The bcd algorithm here is compatible with Ding's code
'''

def setup_plot_subproblem(problem_name):
    output_dir = f'bcd_within_ding_plots_{problem_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return output_dir

def estimate_coefficients(f, theta, j, sigma, key, theta_vals):
    fun_vals = []
    for val in theta_vals:
        key, subkey = jrd.split(key)
        fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
        fun_vals.append(fun_val)
    return np.array(fun_vals)

def interp_matrix(theta_vals):
    # create interpolation matrix
    return np.array([[1, np.cos(val), np.sin(val)] for val in theta_vals])

def update_theta(opt_goal, a, b, c, theta, j, appr_single_var_fun):
    # The goal here is to find the analytic solution of appr_single_var_fun, not exact_single_var_fun
    # And the solution should be within 0 to 2*pi
    if np.isclose(b, 0) and np.isclose(c, 0):
        # Constant function. Any value is an extrema, so it remains unchanged.
        pass
    elif np.isclose(b, 0) and not np.isclose(c, 0):
        # sin function, extrema influenced by amplitude c
        if opt_goal == 'max':
            new_theta_j = (np.pi / 2) if c > 0 else (3 * np.pi / 2)
        elif opt_goal == 'min':
            new_theta_j = (3 * np.pi / 2) if c > 0 else (np.pi / 2)
        theta = theta.at[j].set(new_theta_j)
    elif np.isclose(c, 0) and not np.isclose(b, 0):
        # cos function, extrema influenced by amplitude b
        if opt_goal == 'max':
            new_theta_j = 0.0 if b > 0 else np.pi
        elif opt_goal == 'min':
            new_theta_j = np.pi if b > 0 else 0.0
        theta = theta.at[j].set(new_theta_j)
    else:
        _theta_star = np.arctan(c / b)
        theta = theta.at[j].set(_theta_star)
        IS_MAXIMIZER = appr_single_var_fun(_theta_star) > a
        IS_POSITIVE = _theta_star > 0
        if opt_goal == 'max':
            if IS_POSITIVE:
                if not IS_MAXIMIZER:
                    theta = theta.at[j].add(np.pi)
            else:
                if IS_MAXIMIZER:
                    theta = theta.at[j].add(2 * np.pi)
                else:
                    theta = theta.at[j].add(np.pi)
        elif opt_goal == 'min':
            if IS_POSITIVE:
                if IS_MAXIMIZER:
                    theta = theta.at[j].add(np.pi)
            else:
                if IS_MAXIMIZER:
                    theta = theta.at[j].add(np.pi)
                else:
                    theta = theta.at[j].add(2 * np.pi)
    return theta

def block_coordinate_descent(f, initial_point, num_iterations, sigma, key,
                             problem_name, 
                             opt_goal='max', 
                             plot_subproblem=False,
                             cyclic_mode=False, 
                             fevl_num_each_iter=6,
                             # mode='classical', 'general', 'opt_rcd', 'reg'
                             mode='classical',
                             alpha=0.8):
    
    theta = initial_point
    best_point = theta
    best_value = f(theta)
    function_values = [best_value]

    if plot_subproblem:
        output_dir = setup_plot_subproblem(problem_name)

    print("-"*100)
    
    t = trange(num_iterations, desc="Bar desc", leave=True)
    m = len(theta)
    for i in t:
        if cyclic_mode:
            j = i % m
        else:
            key, subkey = jrd.split(key)
            j = jrd.randint(subkey, shape=(), minval=0, maxval=m)

        theta_old = theta.copy()

        if mode == 'classical':
            # our first version of the algorithm
            theta_vals = [0, np.pi/2, np.pi]
            # we give the explicit form of the inverse of matrix A
            inv_A = np.array([
                    [1/2, 0, 1/2],
                    [1/2, 0, -1/2],
                    [-1/2, 1, -1/2]
                    ])
            fun_vals = estimate_coefficients(f, theta, j, sigma, key, theta_vals)
            a, b, c = inv_A @ fun_vals
        elif mode == 'random_thetas':
            theta_vals = np.random.uniform(0, 2 * np.pi, size=3)
            A = interp_matrix(theta_vals)
            fun_vals = estimate_coefficients(f, theta, j, sigma, key, theta_vals)
            a, b, c = np.linalg.solve(A, fun_vals)
        elif mode == 'general' or mode == 'opt_rcd' or mode == 'opt_rcd2':
            theta_vals = [0, np.pi*2/3, np.pi*4/3]
            inv_A = np.array([
                    [1/3, 1/3, 1/3],
                    [2/3, -1/3, -1/3],
                    [0, 1/np.sqrt(3), -1/np.sqrt(3)]
                    ])
            fun_vals = estimate_coefficients(f, theta, j, sigma, key, theta_vals)
            a, b, c = inv_A @ fun_vals
        elif mode == 'reg':
            theta_vals = np.linspace(0, 2 * np.pi, fevl_num_each_iter, endpoint=False)
            A_reg = interp_matrix(theta_vals) # tall matrix
            fun_vals = estimate_coefficients(f, theta, j, sigma, key, theta_vals)
            solution = np.linalg.lstsq(A_reg, fun_vals, rcond=None) # TODO 有闭式解
            a, b, c = solution[0]

        def appr_single_var_fun(x):
            return a + b * np.cos(x) + c * np.sin(x)
        
        def appr_f_prime(x):
            return - b * np.sin(x) + c * np.cos(x)
        
        if mode == 'opt_rcd':
            # 使用 appr fun 估计偏导，然后更新
            decay_step=30
            decay_rate=0.85
            decay_threshold=1e-4
            # alpha=1.0
            if decay_rate > 0 and (i + 1 ) % decay_step == 0:
                alpha = alpha * decay_rate
                alpha = max(alpha, decay_threshold)
            # learning_rate = 1.0
            learning_rate = alpha
            gradient = appr_f_prime(theta[j])
            sign = 1 if opt_goal == 'max' else -1
            theta = theta.at[j].add(sign * learning_rate * gradient)
        elif mode == 'opt_rcd2':
            # 使用 appr fun 估计偏导，然后更新. 更新2次。
            decay_step=30
            decay_rate=0.85
            decay_threshold=1e-4
            # alpha=1.0
            if decay_rate > 0 and (i + 1 ) % decay_step == 0:
                alpha = alpha * decay_rate
                alpha = max(alpha, decay_threshold)
            # learning_rate = 1.0
            learning_rate = alpha
            sign = 1 if opt_goal == 'max' else -1
            for _ in range(2):
                gradient = appr_f_prime(theta[j])
                theta = theta.at[j].add(sign * learning_rate * gradient)
        else: # mode='classical', 'general', 'reg'
            theta = update_theta(opt_goal, a, b, c, theta, j, appr_single_var_fun)

        next_point = theta
        next_value = f(theta)
        function_values.append(next_value)

        if next_value > best_value:
            best_point = next_point
            best_value = next_value

        message = f"Iteration: {i}, Value: {next_value}, Coord j: {j}({m})"
        t.set_description(f"[BCD-{mode}] Processing %s" % message)
        t.refresh()

        if plot_subproblem:
            def exact_single_var_fun(theta_j):
                return f(theta_old.at[j].set(theta_j))

            x = np.linspace(0, 2 * np.pi, 500)
            exact_y = np.array([exact_single_var_fun(value) for value in x])
            appr_y = np.array([appr_single_var_fun(value) for value in x])
            plt.plot(x, exact_y, label='exact f(theta_j)')
            plt.plot(x, appr_y, label='appr f(theta_j)', linestyle='--')

            old_theta_j = theta_old[j]
            new_theta_j = theta[j]
            theta_j_x = np.array([old_theta_j, new_theta_j])
            exact_theta_j_y = np.array([exact_single_var_fun(value) for value in theta_j_x])
            appr_theta_j_y = np.array([appr_single_var_fun(value) for value in theta_j_x])

            plt.scatter(theta_j_x[0], exact_theta_j_y[0], color='red', s=100, label='Old theta_j')
            plt.scatter(theta_j_x[1], exact_theta_j_y[1], color='blue', s=100, label='New theta_j')
            plt.scatter(theta_j_x[0], appr_theta_j_y[0], color='red', s=100)
            plt.scatter(theta_j_x[1], appr_theta_j_y[1], color='blue', s=100)

            plt.legend()
            plt.xlabel('theta_j')
            plt.ylabel('f(theta_j)')

            # Check if the iteration has worsened and update the title accordingly
            title = f'Iter # {i}, Chosen Coord j: {j}/({m}), Model: {mode}'
            if opt_goal == 'max' and exact_theta_j_y[0] > exact_theta_j_y[1]:
                title += ' (Iteration Worsened!)'
            elif opt_goal == 'min' and exact_theta_j_y[0] < exact_theta_j_y[1]:
                title += ' (Iteration Worsened!)'

            plt.title(title)
            plt.savefig(f'{output_dir}/{problem_name}_iter_{i}_cor_{j}.png')
            plt.close()


    # print(f"[BCD-{mode}] Final value of x:", next_point)
    print(f"[BCD-{mode}] Final value of f:", best_value)

    return best_point, best_value, function_values

