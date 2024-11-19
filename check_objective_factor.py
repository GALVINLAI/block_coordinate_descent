import argparse
import os
import matplotlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
# from jax.config import config
from jax.scipy.linalg import expm
from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from utils import dump, make_dir, hamiltonian_to_matrix

# from algo.bcd import block_coordinate_descent
# Set up configurations
# config.update("jax_enable_x64", True)
# matplotlib.use("Agg")  # Set the matplotlib backend to 'Agg'
np.random.seed(42)

# Number of qubits in the system
n_qubits = 4

# Adding the configuration here
def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add the system size argument
    parser.add_argument('--N', type=int, default=4, help='System size')

    # Add the problem dimension argument
    parser.add_argument('--dim', type=int, default=20, 
                        help='The dimension of the problem')
    
    # Add the sigma argument
    parser.add_argument('--sigma', type=float, default=0.1, 
                        help='The sigma for the Gaussian noise of the gradient. Note that this is multiplied by the gradient')
    
    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=10, 
                        help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.18, 
                        help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.18, 
                        help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=1000, 
                        help='The number of iterations for the optimization algorithm')

    return parser

args = create_parser().parse_args()
print("Run the QAOA algorithm for the QUBO factor model")
print(f"System size: {args.N}")
print(f"Problem dimension: {args.dim}")
print(f"Sigma: {args.sigma}")
print(f"Repeat count: {args.repeat}")
print(f"Gradient descent learning rate: {args.lr_gd}")
print(f"Random coordinate descent learning rate: {args.lr_rcd}")
print(f"Number of iterations: {args.num_iter}")

N = args.N
dim = args.dim
sigma = args.sigma
repeat = args.repeat
lr_gd = args.lr_gd
lr_rcd = args.lr_rcd
num_iter = args.num_iter

hamiltonian_str = "-3.0 + 0.5 * Z0 + 0.25 * Z1 + 0.25 * Z2 + 0.5 * Z3 + 0.75 * Z0*Z2 - 0.25 * Z1*Z2 + 0.25 * Z0*Z1 + 0.25 * Z0*Z3 + 0.75 * Z1*Z3 + 0.25 * Z2*Z3 - 0.25 * Z0*Z1*Z2 - 0.25 * Z1*Z2*Z3"
H = hamiltonian_to_matrix(hamiltonian_str)

# Check the ground state
for x in np.linalg.eigh(H)[1][:, :2].T:
    print(np.nonzero(x))
print(int('0110', 2), int('1001', 2))

E_gs = np.linalg.eigh(H)[0][0]

# Create total X operator by applying X gate to each qubit
hamiltonian_str = '+ '.join([f'X{i} ' for i in range(n_qubits)])
total_X_operator = hamiltonian_to_matrix(hamiltonian_str)

# Initial state
psi0 = np.ones(2 ** 4) / 2 ** 2

def qaoa_ansatz(params):
    psi = psi0
    for i, param in enumerate(params): 
        if i % 2 == 0:
            # TODO: Understand the design principle of the FACTOR problem
            # Note!! Here the H in the circuit is not unitary, so it is not involutary!!!
            # So the single variable of the factor problem objective function is not a standard trigonometric function
            psi = expm(-1j * param * H) @ psi
        else: 
            # Note!! Here the total_X_operator in the circuit is not unitary, so it is not involutary!!!
            psi = expm(-1j * param * total_X_operator) @ psi
    return psi

def get_energy_ratio(psi):
    return jnp.vdot(psi, jnp.dot(H, psi)).real / E_gs

@jax.jit
def objective(params):
    psi = qaoa_ansatz(params)
    return get_energy_ratio(psi)

num_layer = 20

dim = 2 * num_layer

params_dryrun = jnp.array([2 * np.pi] * dim)
print(objective(params_dryrun))



ry_angles = params_dryrun


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_single_objective(objective, params, j, plot_flag=True):
    """
    Plots the objective function (e.g., fidelity) as a function of the varied angle
    at a specific index, while keeping other angles fixed. Optionally, disables plotting.

    Arguments:
        objective -- Function handle for the reward (fidelity) function
        params -- List of angles (alpha and beta values) for the params
        j -- Index of the angle to keep fixed
        plot_flag -- Boolean flag to enable or disable plotting (default is True)
    """
    # Define the angle range for the varied angle
    angle_range = np.linspace(-2 * np.pi, 2 * np.pi, 500)  # Angle range from -4π to 4π
    
    # Define the new objective function that varies the angle at position j
    def single_f(x, params):
        params = params.at[j].set(x)
        return objective(params)
    # single_f = lambda x: objective(params[:j] + [x] + params[j+1:])

    # Array to store the objective function values
    single_f_values = []

    # Loop through the angle range and calculate the objective function value for each angle
    for angle in angle_range:
        single_f_values.append(single_f(angle, params))  # Calculate the objective function value

    # If plot_flag is True, plot the objective function
    if plot_flag:
        plt.plot(angle_range, single_f_values)
        plt.xlabel(f'Angle at index {j} (params[{j}])')
        plt.ylabel('Objective Function Value')
        plt.title(f'Objective Function vs Angle at Index {j}')
        plt.grid(True)
        plt.show()

    # Find the peaks (local maxima) in the objective function values
    peaks, _ = find_peaks(single_f_values)

    print("所有的峰值点:", angle_range[peaks])
    
    # If there are peaks, find the maximum peak values
    if len(peaks) > 0:
        peak_values = np.array(single_f_values)[peaks]  # Peak values at the corresponding indices
        max_peak_value = np.max(peak_values)  # Find the maximum peak value

        # Use np.isclose to allow comparison with tolerance (allowing for small errors)
        max_peak_indices = peaks[np.isclose(peak_values, max_peak_value, atol=1e-5)]  # Find indices within tolerance

        print("最大峰值:", max_peak_value)
        print("最大峰值所在的角度位置:", angle_range[max_peak_indices])

        # Find the distances between adjacent maximum peak indices
        if len(max_peak_indices) > 1:
            peak_distances = np.diff(angle_range[max_peak_indices])  # Calculate the distances between adjacent peaks
            print("相邻最大峰值之间的距离:", peak_distances)
        else:
            print("没有找到相邻的最大峰值")
    else:
        print("没有找到峰值")

# # Example usage:
# # Assume `objective` is your function and `ry_angles` is your parameter list
plot_single_objective(objective, ry_angles, 36, plot_flag=True)  # To plot the graph
# plot_single_objective(objective, ry_angles, 2, plot_flag=False)  # To only calculate and print peaks without plotting

# # Loop through all positions in the params array
# for j in range(len(ry_angles)):
#     print(f"Processing angle at index {j} (params[{j}])")
#     plot_single_objective(objective, ry_angles, j, plot_flag=False)


