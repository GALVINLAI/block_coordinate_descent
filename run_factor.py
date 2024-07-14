# The code is adapted from https://minatoyuichiro.medium.com/variational-quantum-factoring-using-qaoa-and-vqe-on-blueqat-29c6f4f195f1 and https://arxiv.org/pdf/1411.6758.pdf
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

from algo.bcd import block_coordinate_descent
# Set up configurations
# config.update("jax_enable_x64", True)
matplotlib.use("Agg")  # Set the matplotlib backend to 'Agg'
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

def main():
    # Try the gradient descent algorithm
    
    make_dir('exp/factor')
    make_dir(f'exp/factor/dim_{dim}')
    random_keys = jrd.split(jrd.PRNGKey(42), repeat)
    for exp_i in range(repeat):
        # Define the initial value for x
        x = jrd.uniform(random_keys[exp_i], (dim, )) * 2 * np.pi
        
        if os.path.exists(f'exp/factor/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl'):
            continue

        # # Run gradient descent
        # function_values_gd, x_gd, f_x_gd, eigen_values_gd, lip_diag_values_gd = gradient_descent(objective, x, lr_gd, num_iter, sigma, random_keys[exp_i], skip_hessian=True)

        # # Run random coordinate descent
        # x_rcd, f_x_rcd, function_values_rcd, eigen_values_rcd, lip_diag_values_rcd = random_coordinate_descent(
        #     objective, x, lr_rcd, num_iter, sigma, random_keys[exp_i], skip_hessian=True
        # )

        MAX_ITER_BCD = 100
        # Run block coordinate descent
        x_bcd, f_x_bcd, function_values_bcd, eigen_values_bcd, lip_diag_values_bcd = block_coordinate_descent(
            objective, x, num_iter, sigma, random_keys[exp_i],
            PLOT_SUBPROBLEM=True
        )

        make_dir(f'exp/factor/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}')
        data_dict = {
            'function_values_gd': function_values_gd,
            'function_values_rcd': function_values_rcd,
            'function_values_bcd': function_values_bcd,
            'x_gd': x_gd,
            'x_rcd': x_rcd,
            'x_bcd': x_bcd,
            'f_x_gd': f_x_gd,
            'f_x_rcd': f_x_rcd,
            'f_x_bcd': f_x_bcd,
            'eigen_values_gd': eigen_values_gd,
            'eigen_values_rcd': eigen_values_rcd,
            'eigen_values_bcd': eigen_values_bcd,
            'lip_diag_values_gd': lip_diag_values_gd,
            'lip_diag_values_rcd': lip_diag_values_rcd,
            'lip_diag_values_bcd': lip_diag_values_bcd,
        }

        dump(data_dict, f'exp/factor/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl')

if __name__ == "__main__":
    main()
