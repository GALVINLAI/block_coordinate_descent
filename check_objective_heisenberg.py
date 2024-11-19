import argparse
import os
import sys
import matplotlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.random as jrd
# from jax.config import config
from utils import load, dump, make_dir

from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
# from algo.bcd import block_coordinate_descent

# Set up configurations
np.random.seed(42)
# config.update("jax_enable_x64", True)

# Set the matplotlib backend to 'Agg'
# matplotlib.use("Agg")

def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add the system size argument
    parser.add_argument('--N', type=int, default=4, help='System size')

    # Add the problem dimension argument
    parser.add_argument('--dim', type=int, default=6, 
                        help='The dimension of the problem')
    
    # Add the sigma argument
    parser.add_argument('--sigma', type=float, default=0.01, 
                        help='The sigma for the Gaussian noise of the gradient. Note that this is multiplied by the gradient')
    
    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=10, 
                        help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.001, 
                        help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.01, 
                        help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=10000, 
                    help='The number of iterations for the optimization algorithm')

    return parser

args = create_parser().parse_args()
print("Run the QAOA algorithm for the Heisenberg model")
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

# Load the data from the pre-dumped data
try:
    data_dict = load(f"quspin_data/heisenberg_N_{N}.pkl")
except FileNotFoundError:
    print("Unable to load the data file. Please run the code `python generate_heisenberg_ham.py` to generate the data.")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit()

psi0_input = data_dict["psi0_input"]
psi1_input = data_dict["psi1_input"]
H0 = data_dict["H0"]
H1 = data_dict["H1"]

# Convert the data to jax array
psi0_input = jnp.array(psi0_input)
psi1_input = jnp.array(psi1_input)
H0 = jnp.array(H0)
H1 = jnp.array(H1)

# Get the eigenvalues and eigenvectors
H0_eval, H0_evec = jla.eigh(H0)
H1_eval, H1_evec = jla.eigh(H1)
imag_unit = jnp.complex64(1.0j)

def get_reward(protocol):
    """Get the fidelity of the protocol
    Arguments:
        protocol -- The alpha's and beta's for a given protocol
    Returns:
        fidelity -- scalar between 0 and 1
    """
    u = psi0_input
    for i in range(len(protocol)):
        if i % 2 == 0:
            u = jnp.matmul(H0_evec.conj().T, u)
            u = jnp.multiply(jnp.exp(-protocol[i] * imag_unit * H0_eval), u)
            u = jnp.matmul(H0_evec, u)
        else:
            u = jnp.matmul(H1_evec.conj().T, u)
            u = jnp.multiply(jnp.exp(-protocol[i] * imag_unit * H1_eval), u)
            u = jnp.matmul(H1_evec, u)

    return jnp.absolute(jnp.dot(psi1_input.T.conjugate(), u)) ** 2


# define the rotation angles for the RY gates
ry_angles = [jnp.pi/4] * dim  # using pi/4 for each qubit as an example

objective = get_reward

print(objective(ry_angles))









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
    angle_range = np.linspace(- 4*np.pi,  4*np.pi, 2000)  # Angle range from -4π to 4π

    # Define the new objective function that varies the angle at position j
    single_f = lambda x: objective(params[:j] + [x] + params[j+1:])

    # Array to store the objective function values
    single_f_values = []

    # Loop through the angle range and calculate the objective function value for each angle
    for angle in angle_range:
        single_f_values.append(single_f(angle))  # Calculate the objective function value

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
# plot_single_objective(objective, ry_angles, 0, plot_flag=True)  # To plot the graph
# plot_single_objective(objective, ry_angles, 2, plot_flag=False)  # To only calculate and print peaks without plotting

# Loop through all positions in the params array
for j in range(len(ry_angles)):
    print(f"Processing angle at index {j} (params[{j}])")
    plot_single_objective(objective, ry_angles, j, plot_flag=True)


