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

def block_coordinate_descent_c(f, initial_point, num_iterations, sigma, key,
                             problem_name,
                             opt_goal='max',
                             plot_subproblem=False,
                             cyclic_mode=False):
    

    theta = initial_point
    best_point = theta
    best_value = f(theta)
    function_values = [best_value]
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

        #####################################################################
        # Estimate coefficients a, b, c (with noise) by [0, np.pi / 2, np.pi]
        theta_vals = [0, np.pi / 2, np.pi]
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        hat_f1, hat_f2, hat_f3 = fun_vals
        a_c = (hat_f1 + hat_f3) / 2
        b_c = (hat_f1 - hat_f3) / 2
        c_c = hat_f2 - a_c 
        def appr_single_var_fun_classical(theta_j):
            return a_c + b_c * np.cos(theta_j) + c_c * np.sin(theta_j)


        #####################################################################
        # 使用其他点做三角差值获得 appr_single_var_fun
        theta_vals = [5.390303024888212, 1.2015025457884192, 3.2959214465226303]
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        fun_vals = np.array(fun_vals)
        def create_matrix(theta_vals):
            return np.array([
                [1, np.cos(theta_vals[0]), np.sin(theta_vals[0])],
                [1, np.cos(theta_vals[1]), np.sin(theta_vals[1])],
                [1, np.cos(theta_vals[2]), np.sin(theta_vals[2])]
            ])
        A_opt = create_matrix(theta_vals)
        a_g, b_g, c_g = np.linalg.solve(A_opt, fun_vals)
        def appr_single_var_fun_general(theta_j):
            return a_g + b_g * np.cos(theta_j) + c_g * np.sin(theta_j)
        # 当sigma=0, abc_c = abc_g


        # if opt_method == 'solver':
        #     raise ValueError('Do not use solvers! This is debatable.')
        # elif opt_method == 'analytic':
            
        a = a_c
        b = b_c
        c = c_c

        # 具体迭代计算让 appr_single_var_fun_classical 来做

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

            IS_MAXIMIZER = appr_single_var_fun_classical(_theta_star) > a
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
        # print(f"\n a: {a}, b: {b}, c: {c}")
        t.set_description("[BCD-classical] Processing %s" % message)
        t.refresh()

        if plot_subproblem:
            def exact_single_var_fun(theta_j):
                return f(theta_old.at[j].set(theta_j))

            def plot_single_var_fun():
                x_range = 4 * np.pi
                x = np.linspace(-x_range, x_range, 500)
                x = np.linspace(0, 2*np.pi, 500)

                exact_y = np.array([exact_single_var_fun(value) for value in x])
                appr_classical_y = np.array([appr_single_var_fun_classical(value) for value in x])
                appr_general_y = np.array([appr_single_var_fun_general(value) for value in x])

                plt.plot(x, exact_y, label='exact f(theta_j)')
                plt.plot(x, appr_classical_y, label='appr classical f(theta_j)', linestyle='--')
                plt.plot(x, appr_general_y, label='appr general f(theta_j)', linestyle='dotted')

                old_theta_j = theta_old[j]
                new_theta_j = theta[j]
                theta_j_x = np.array([old_theta_j, new_theta_j])

                exact_theta_j_y = np.array([exact_single_var_fun(value) for value in theta_j_x])

                appr_theta_j_y = np.array([appr_single_var_fun_classical(value) for value in theta_j_x])
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

    print("[BCD-classical] Final value of x:", next_point)
    print("[BCD-classical] Final value of f:", best_value)

    return best_point, best_value, function_values



def block_coordinate_descent_g(f, initial_point, num_iterations, sigma, key,
                             problem_name,
                             opt_goal='max', 
                             plot_subproblem=False,
                             cyclic_mode=False):
    
    
    

    theta = initial_point
    best_point = theta
    best_value = f(theta)

    function_values = [best_value]
    
    
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





        #####################################################################
        # Estimate coefficients a, b, c (with noise) by [0, np.pi / 2, np.pi]
        theta_vals = [0, np.pi / 2, np.pi]
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        hat_f1, hat_f2, hat_f3 = fun_vals
        a_c = (hat_f1 + hat_f3) / 2
        b_c = (hat_f1 - hat_f3) / 2
        c_c = hat_f2 - a_c 
        def appr_single_var_fun_classical(theta_j):
            return a_c + b_c * np.cos(theta_j) + c_c * np.sin(theta_j)


        #####################################################################
        # 使用其他点做三角差值获得 appr_single_var_fun
        theta_vals = [0, 2 * np.pi/ 3 , 4 * np.pi/ 3 ]
      
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        fun_vals = np.array(fun_vals)
        def create_matrix(theta_vals):
            return np.array([
                [1, np.cos(theta_vals[0]), np.sin(theta_vals[0])],
                [1, np.cos(theta_vals[1]), np.sin(theta_vals[1])],
                [1, np.cos(theta_vals[2]), np.sin(theta_vals[2])]
            ])
        A_opt = create_matrix(theta_vals)
        a_g, b_g, c_g = np.linalg.solve(A_opt, fun_vals)
        def appr_single_var_fun_general(theta_j):
            return a_g + b_g * np.cos(theta_j) + c_g * np.sin(theta_j)
        # 当sigma=0, abc_c = abc_g

        # if opt_method == 'solver':
        #     raise ValueError('Do not use solvers! This is debatable.')
        # elif opt_method == 'analytic':
            
        a = a_g
        b = b_g
        c = c_g

        # 具体迭代计算让 appr_single_var_fun_classical 来做

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

            IS_MAXIMIZER = appr_single_var_fun_general(_theta_star) > a
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
        # print(f"\n a: {a}, b: {b}, c: {c}")
        t.set_description("[BCD-general] Processing %s" % message)
        t.refresh()

        if plot_subproblem:
            def exact_single_var_fun(theta_j):
                return f(theta_old.at[j].set(theta_j))

            def plot_single_var_fun():
                x_range = 4 * np.pi
                x = np.linspace(-x_range, x_range, 500)
                x = np.linspace(0, 2*np.pi, 500)

                exact_y = np.array([exact_single_var_fun(value) for value in x])
                appr_classical_y = np.array([appr_single_var_fun_classical(value) for value in x])
                appr_general_y = np.array([appr_single_var_fun_general(value) for value in x])

                plt.plot(x, exact_y, label='exact f(theta_j)')
                plt.plot(x, appr_classical_y, label='appr classical f(theta_j)', linestyle='--')
                plt.plot(x, appr_general_y, label='appr general f(theta_j)', linestyle='dotted')

                old_theta_j = theta_old[j]
                new_theta_j = theta[j]
                theta_j_x = np.array([old_theta_j, new_theta_j])

                exact_theta_j_y = np.array([exact_single_var_fun(value) for value in theta_j_x])

                appr_theta_j_y = np.array([appr_single_var_fun_classical(value) for value in theta_j_x])
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

    print("[BCD-general] Final value of x:", next_point)
    print("[BCD-general] Final value of f:", best_value)

    return best_point, best_value, function_values




def block_coordinate_descent_reg(f, initial_point, num_iterations, sigma, key,
                             problem_name,
                             opt_goal='max', 
                             plot_subproblem=False,
                             cyclic_mode=False,
                             fevl_num_each_iter=3):
    

    theta = initial_point
    best_point = theta
    best_value = f(theta)

    function_values = [best_value]
    
    
    analytic_solved_flags = []

    if plot_subproblem:
        output_dir = f'bcd_within_ding_plots_{problem_name}'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    points_num = fevl_num_each_iter

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


    


        #####################################################################
        # Estimate coefficients a, b, c (with noise) by [0, np.pi / 2, np.pi]
        theta_vals = [0, np.pi / 2, np.pi]
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        hat_f1, hat_f2, hat_f3 = fun_vals
        a_c = (hat_f1 + hat_f3) / 2
        b_c = (hat_f1 - hat_f3) / 2
        c_c = hat_f2 - a_c 
        def appr_single_var_fun_classical(theta_j):
            return a_c + b_c * np.cos(theta_j) + c_c * np.sin(theta_j)


        #####################################################################
        # 使用其他点做三角差值获得 appr_single_var_fun
        # theta_vals = [5.390303024888212, 1.2015025457884192, 3.2959214465226303]
        # theta_vals = [5.62777484, 1.43898474, 3.53338003]
        # theta_vals = [0.0, 2 * np.pi/ 3 , 4 * np.pi/ 3 ]
        
        # if i > 0 and (i + 1 ) % 30 == 0:
        #     points_num += 1

        theta_vals = np.linspace(0, 2 * np.pi, points_num, endpoint=False)
        theta_vals = theta_vals.tolist()
        # theta_vals = np.random.uniform(0, 2 * np.pi, size=(3,))
        # theta_vals = [0.16521743513397463, 2.2596095415271282, 4.354009870477592]
        
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        fun_vals = np.array(fun_vals)
        
        def create_matrix(theta_vals):
            matrix = []
            for theta in theta_vals:
                matrix.append([1, np.cos(theta), np.sin(theta)])
            return np.array(matrix)
        
        A_reg = create_matrix(theta_vals)
        solution = np.linalg.lstsq(A_reg, fun_vals, rcond=None)
        a_reg, b_reg, c_reg = solution[0]

        # a_g, b_g, c_g = np.linalg.solve(A_opt, fun_vals)

        def appr_single_var_fun_reg(theta_j):
            return a_reg + b_reg * np.cos(theta_j) + c_reg * np.sin(theta_j)
        # 当sigma=0, abc_c = abc_g


        # if opt_method == 'solver':
        #     raise ValueError('Do not use solvers! This is debatable.')
        # elif opt_method == 'analytic':
            
        a = a_reg
        b = b_reg
        c = c_reg

        # 具体迭代计算让 appr_single_var_fun_classical 来做

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

            IS_MAXIMIZER = appr_single_var_fun_reg(_theta_star) > a
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
        # print(f"\n a: {a}, b: {b}, c: {c}")
        t.set_description("[BCD-reg] Processing %s" % message)
        t.refresh()


        if plot_subproblem:
            def exact_single_var_fun(theta_j):
                return f(theta_old.at[j].set(theta_j))

            def plot_single_var_fun():
                x_range = 4 * np.pi
                x = np.linspace(-x_range, x_range, 500)
                x = np.linspace(0, 2*np.pi, 500)

                exact_y = np.array([exact_single_var_fun(value) for value in x])
                appr_classical_y = np.array([appr_single_var_fun_classical(value) for value in x])
                appr_reg_y = np.array([appr_single_var_fun_reg(value) for value in x])

                plt.plot(x, exact_y, label='exact f(theta_j)')
                plt.plot(x, appr_classical_y, label='appr classical f(theta_j)', linestyle='--')
                plt.plot(x, appr_reg_y, label='appr reg f(theta_j)', linestyle='dotted')

                old_theta_j = theta_old[j]
                new_theta_j = theta[j]
                theta_j_x = np.array([old_theta_j, new_theta_j])

                exact_theta_j_y = np.array([exact_single_var_fun(value) for value in theta_j_x])

                appr_theta_j_y = np.array([appr_single_var_fun_classical(value) for value in theta_j_x])
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

    print("[BCD-reg] Final value of x:", next_point)
    print("[BCD-reg] Final value of f:", best_value)

    return best_point, best_value, function_values


def block_coordinate_descent_opt_rcd(f, initial_point, num_iterations, sigma, key,
                             problem_name,
                             opt_goal='max',
                             plot_subproblem=False,
                             cyclic_mode=False,
                             alpha=1.0):
    
    
    

    theta = initial_point
    best_point = theta
    best_value = f(theta)

    function_values = [best_value]
    
    
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





        #####################################################################
        # Estimate coefficients a, b, c (with noise) by [0, np.pi / 2, np.pi]
        theta_vals = [0, np.pi / 2, np.pi]
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        hat_f1, hat_f2, hat_f3 = fun_vals
        a_c = (hat_f1 + hat_f3) / 2
        b_c = (hat_f1 - hat_f3) / 2
        c_c = hat_f2 - a_c 
        def appr_single_var_fun_classical(theta_j):
            return a_c + b_c * np.cos(theta_j) + c_c * np.sin(theta_j)


        #####################################################################
        # 使用其他点做三角差值获得 appr_single_var_fun
        # theta_vals = [5.390303024888212, 1.2015025457884192, 3.2959214465226303]
        # theta_vals = [5.62777484, 1.43898474, 3.53338003]
        theta_vals = [0, 2 * np.pi/ 3 , 4 * np.pi/ 3 ]
        # theta_vals = np.random.uniform(0, 2 * np.pi, size=(3,))
        # theta_vals = [0.16521743513397463, 2.2596095415271282, 4.354009870477592]
        
        fun_vals = []
        for val in theta_vals:
            key, subkey = jrd.split(key)
            fun_val = f(theta.at[j].set(val)) + jrd.normal(subkey, shape=()) * sigma
            fun_vals.append(fun_val)
        fun_vals = np.array(fun_vals)
        def create_matrix(theta_vals):
            return np.array([
                [1, np.cos(theta_vals[0]), np.sin(theta_vals[0])],
                [1, np.cos(theta_vals[1]), np.sin(theta_vals[1])],
                [1, np.cos(theta_vals[2]), np.sin(theta_vals[2])]
            ])
        A_opt = create_matrix(theta_vals)
        a_g, b_g, c_g = np.linalg.solve(A_opt, fun_vals)
        def appr_single_var_fun_general(theta_j):
            return a_g + b_g * np.cos(theta_j) + c_g * np.sin(theta_j)
        # 当sigma=0, abc_c = abc_g

        def appr_f_prime(theta_j):
            return -b_g * np.sin(theta_j) + c_g * np.cos(theta_j)
        
        decay_step=30
        decay_rate=-1
        decay_threshold=1e-4
        if decay_rate > 0 and (i + 1 ) % decay_step == 0:
            alpha = alpha * decay_rate
            alpha = max(alpha, decay_threshold)

        # learning_rate=1.0
        learning_rate=1.0

        gradient = appr_f_prime(theta[j])
        if opt_goal == 'max':
            theta = theta.at[j].add(learning_rate * gradient)
        elif opt_goal == 'min':
            theta = theta.at[j].add(-learning_rate * gradient)

        next_point = theta
        next_value = f(theta)
        function_values.append(next_value)
        # analytic_solved_flags.append(solved_flag)

        if next_value > best_value:
            best_point = next_point
            best_value = next_value

        message = f"Iteration: {i}, Value: {next_value}, Coord j: {j}({m})"
        # print(f"\n a: {a}, b: {b}, c: {c}")
        t.set_description("[BCD-general] Processing %s" % message)
        t.refresh()


        if plot_subproblem:
            def exact_single_var_fun(theta_j):
                return f(theta_old.at[j].set(theta_j))

            def plot_single_var_fun():
                x_range = 4 * np.pi
                x = np.linspace(-x_range, x_range, 500)
                x = np.linspace(0, 2*np.pi, 500)

                exact_y = np.array([exact_single_var_fun(value) for value in x])
                appr_classical_y = np.array([appr_single_var_fun_classical(value) for value in x])
                appr_general_y = np.array([appr_single_var_fun_general(value) for value in x])

                plt.plot(x, exact_y, label='exact f(theta_j)')
                plt.plot(x, appr_classical_y, label='appr classical f(theta_j)', linestyle='--')
                plt.plot(x, appr_general_y, label='appr general f(theta_j)', linestyle='dotted')

                old_theta_j = theta_old[j]
                new_theta_j = theta[j]
                theta_j_x = np.array([old_theta_j, new_theta_j])

                exact_theta_j_y = np.array([exact_single_var_fun(value) for value in theta_j_x])

                appr_theta_j_y = np.array([appr_single_var_fun_classical(value) for value in theta_j_x])
                plt.scatter(theta_j_x[0], exact_theta_j_y[0], color='red', s=100, label='Old theta_j')
                plt.scatter(theta_j_x[1], exact_theta_j_y[1], color='blue', s=100, label='New theta_j')
                plt.scatter(theta_j_x[0], appr_theta_j_y[0], color='red', s=100)
                plt.scatter(theta_j_x[1], appr_theta_j_y[1], color='blue', s=100)
                plt.legend()
                plt.xlabel('theta_j')
                plt.ylabel('f(theta_j)')
                plt.title(f'Iter # {i}, Chosen Coord j: {j}/({m}), Range of x: [0, 2*pi]')
                plt.savefig(f'{output_dir}/{problem_name}_iter_{i}_cor_{j}.png')
                plt.close()

            plot_single_var_fun()

    print("[BCD-general] Final value of x:", next_point)
    print("[BCD-general] Final value of f:", best_value)

    return best_point, best_value, function_values