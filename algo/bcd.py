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

def block_coordinate_descent(f, initial_point, num_iterations, sigma, key,
                             problem_name,
                             opt_goal='max', 
                             opt_method='analytic',
                             skip_hessian=False, 
                             plot_subproblem=False,
                             cyclic_mode=False):
    
    grad_f = jax.jit(jax.grad(f))
    hess_f = jax.jit(jax.hessian(f))

    theta = initial_point
    best_point = theta
    best_value = f(theta)

    function_values = [best_value]
    eigen_values = []
    lip_diag_values = []
    analytic_solved_flags = []

    if plot_subproblem:
        output_dir = f'bcd_within_ding_plots_{problem_name}'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # Print a separator before the progress bar
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

        if opt_method == 'solver':
            raise ValueError('Do not use solvers! This is debatable.')
        elif opt_method == 'analytic':
            # Estimate coefficients a, b, c (with noise)
            theta_vals = [0, np.pi / 2, np.pi]
            fun_vals = []
            for val in theta_vals:
                key, subkey = jrd.split(key)
                fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
                fun_vals.append(fun_val)
            hat_f1, hat_f2, hat_f3 = fun_vals
           
            a = (hat_f1 + hat_f3) / 2
            b = (hat_f1 - hat_f3) / 2
            c = hat_f2 - a 

            def appr_single_var_fun(theta_j):
                return a + b * np.cos(theta_j) + c * np.sin(theta_j)

            # The goal here is to find the analytic solution of appr_single_var_fun, not exact_single_var_fun
            # And the solution should be within 0 to 2*pi
            if np.isclose(b, 0) and np.isclose(c, 0):
                # Constant function. Any value is an extrema, so it remains unchanged.
                solved_flag = 1
                pass
            elif np.isclose(b, 0) and not np.isclose(c, 0):
                # sin function, extrema influenced by amplitude c
                solved_flag = 2
                if opt_goal == 'max':
                    new_theta_j = (np.pi / 2) if c > 0 else (3 * np.pi / 2)
                elif opt_goal == 'min':
                    new_theta_j = (3 * np.pi / 2) if c > 0 else (np.pi / 2)
                theta = theta.at[j].set(new_theta_j)   
            elif np.isclose(c, 0) and not np.isclose(b, 0):
                # cos function, extrema influenced by amplitude b
                solved_flag = 3
                if opt_goal == 'max':
                    new_theta_j = 0.0 if b > 0 else np.pi
                elif opt_goal == 'min':
                    new_theta_j = np.pi if b > 0 else 0.0
                theta = theta.at[j].set(new_theta_j)
            else:
                # The most complex case
                solved_flag = 4
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
                
        next_point = theta
        next_value = f(theta)
        function_values.append(next_value)
        analytic_solved_flags.append(solved_flag)

        if next_value > best_value:
            best_point = next_point
            best_value = next_value

        message = f"Iteration: {i}, Value: {next_value}, Coord j: {j}({m})"
        print(f"\n a: {a}, b: {b}, c: {c}")
        t.set_description("[BCD] Processing %s" % message)
        t.refresh()

        if not skip_hessian:
            hessian_mat = hess_f(theta)
            vals, _ = np.linalg.eig(np.array(hessian_mat))
            eigen_values.append(vals)
            lip_diag_values.append(np.diag(hessian_mat))

        if plot_subproblem:
            def exact_single_var_fun(theta_j):
                return f(theta_old.at[j].set(theta_j))

            def plot_single_var_fun():
                x_range = 2 * np.pi
                x = np.linspace(0, x_range, 100)
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
                plt.title(f'Iter # {i}, Chosen Coord j: {j}/({m}), Range of x: [0, 2*pi], Flag: {solved_flag}')
                plt.savefig(f'{output_dir}/{problem_name}_iter_{i}_cor_{j}.png')
                plt.close()

            plot_single_var_fun()

    print("[BCD] Final value of x:", next_point)
    print("[BCD] Final value of f:", best_value)

    return best_point, best_value, function_values, eigen_values, lip_diag_values
