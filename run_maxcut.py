import argparse
import os, shutil
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
# from jax.config import config
from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from utils import dump, make_dir, hamiltonian_to_matrix
from algo.bcd import block_coordinate_descent

# Set up configurations
# config.update("jax_enable_x64", True)
np.random.seed(6)

# adding the configuration here
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
    parser.add_argument('--lr_gd', type=float, default=0.1, 
                        help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=1.0, 
                        help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=200, 
                    help='The number of iterations for the optimization algorithm')

    return parser

args = create_parser().parse_args()
print("Run the QAOA algorithm for the maxcut model")
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

######################## max-cut problem setup ########################
# Convert the Hamiltonian in string form to a matrix and verify it.
ham_str = '0.5 - 3 * z0  + 0.5 * z1 * z0 + 0.5 * z2 * z0 + 0.5 * z2 * z1 + 0.5 * z3 * z0 + 0.5 * z3 * z2'
H = hamiltonian_to_matrix(ham_str)


# H=
# Array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0.,  0., -3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0.,  0.,  0., -3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0., -4.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0., -3.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -3.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  4.,  0., 0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 3.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  4.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  6.]], dtype=float32)

# We find that H is indeed a diagonal matrix.
# But the first diagonal element (0) and the last one (6) do not have any practical significance.
# 0 corresponds to 0000, representing the values of x0, x1, x2, x3.
# 16 corresponds to 1111, representing the values of x0, x1, x2, x3.
# These two are not reasonable max-cut solutions

# verify the Hamiltonian
print(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0]))
# The linalg.eigh function returns two arrays: one is the eigenvalue array, the other is the matrix of eigenvectors.
# linalg.eigh(H)[1] extracts the matrix of eigenvectors.
# [:, 0] selects the first eigenvector, which is the ground state eigenvector. Here, [:, 0] means selecting the 0th column of all rows.
# jnp.nonzero(...) returns the indices of the non-zero elements in the input array.

print(int('0101', 2)) # Convert binary string '0101' to decimal integer
# TODO What is this doing?
# 0101 converts to the decimal integer 5
# 0, 1, 0, 1 is the optimal solution, representing the values of x0, x1, x2, x3.
# max-cut is 4.

E_gs = jnp.linalg.eigh(H)[0][0]

def ry(theta):
    """
    Create a rotation matrix for a rotation about the y-axis.
    RY gate
    """
    return jnp.array([[jnp.cos(theta/2), -jnp.sin(theta/2)],
                     [jnp.sin(theta/2), jnp.cos(theta/2)]])

def cz():
    """
    Create a controlled-Z gate.
    """
    return jnp.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def _apply_gate(state, gate, targets, n_qubits):
    """
    Apply a(one) gate to a state.
    The _apply_gate function inside here is obviously very inefficient.
    """
    operators = [jnp.eye(2)] * n_qubits
    # Create a list of n_qubits 2x2 identity matrices.
    # Assuming n_qubits is 4, this line of code is equivalent to: [jnp.eye(2), jnp.eye(2), jnp.eye(2), jnp.eye(2)]

    # Set the specified quantum gate in the corresponding position of the operator list based on the target qubits.
    if len(targets) == 1:
        # If the target qubit is a single qubit (e.g., [1]), place the quantum gate at operators[1].
        operators[targets[0]] = gate
    else:
        # If the target qubit is multiple qubits (e.g., [1, 2]), place the quantum gate in the corresponding range of the operators list.
        operators[targets[0] : targets[1] + 1] = [gate]
    # The rest that has no effect is considered as identity matrices.

    full_gate = operators[0]
    for operator in operators[1:]:
        # Use the Kronecker product to combine single-qubit operators into a complete system operator.
        # jnp.kron calculates the Kronecker product of two matrices.
        full_gate = jnp.kron(full_gate, operator)

    # Apply the calculated complete operator full_gate to the quantum state state.
    return jnp.dot(full_gate, state)

def apply_gates(state, n_qubits, reps, ry_angles):
    """
    Apply the sequence of RY and CZ gates.
    """
    for rep in range(reps):

        for i in range(n_qubits):
            # Apply RY quantum gate to each qubit. _apply_gate applies this RY quantum gate to the i-th qubit.
            # ry(ry_angles[rep * n_qubits + i]) is gate
            # TODO Obviously, the circuit of this max-cut perfectly fits our assumptions
            # So the single variable of the objective function is a standard trigonometric function!!!
            state = _apply_gate(state, ry(ry_angles[rep * n_qubits + i]), [i], n_qubits)

        for i in range(n_qubits - 1):
            # Apply CZ quantum gate to each pair of adjacent qubits.
            state = _apply_gate(state, cz(), [i, i + 1], n_qubits)
            # The _apply_gate function inside here is obviously very inefficient.

    return state

def get_energy_ratio(psi):
    # vdot: Return the dot product of two vectors.
    # Note that E_gs = -4, so we are actually maximizing jnp.vdot(psi, jnp.dot(H, psi)) * (-1 / 4)
    return jnp.vdot(psi, jnp.dot(H, psi)) / E_gs

# define the number of qubits
n_qubits = 4

# define the number of repetitions, which is the number of layers
reps = 5

# dimension of the Hilbert space
dim = n_qubits * reps

# define the rotation angles for the RY gates
ry_angles = [jnp.pi/4] * dim  # using pi/4 for each qubit as an example

# create the initial state |0⟩^⊗n
state = jnp.eye(2**n_qubits)[:, 0]

# initial state
state = apply_gates(state, n_qubits, reps, ry_angles)

# Define the optimization objective function using JAX's jit decorator to accelerate calculations.
@jax.jit
def objective(params):
    # params are ry_angles
    psi = apply_gates(state, n_qubits, reps, params)
    return get_energy_ratio(psi)

# the final state is now stored in the 'state' variable
print(objective(ry_angles))

######################## Solver setup ########################

def main():
    # try the gradient descent algorithm
    
    make_dir('exp/maxcut')
    # make_dir(f'exp/maxcut/dim_{dim}')

    random_keys = jrd.split(jrd.PRNGKey(42), repeat)

    SKIP_ALL_HESSIAN = True

    for exp_i in range(repeat):
        print('='*100)
        print(f'Experiment # {exp_i} begins.')
        # Define the initial value for x
        # Uniformly distributed between [0, 2pi]
        # Ensure that the initial point of each experiment exp_i is random
        x = jrd.uniform(random_keys[exp_i], (dim,)) * 2 * np.pi
        
        # Check if the folder exists and delete it if it does.
        # Note that we delete everything inside the folder
        dir_path = f'exp/maxcut/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}'
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed existing directory: {dir_path}")

        # Check if the file exists and skip if it does.
        # if os.path.exists(f'exp/maxcut/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl'):
        #     continue

        # # Run gradient descent
        # x_gd, f_x_gd, function_values_gd, eigen_values_gd, lip_diag_values_gd = gradient_descent(
        #     objective, x, lr_gd, num_iter, sigma, random_keys[exp_i], 
        #     skip_hessian=SKIP_ALL_HESSIAN
        # )

        # # Run random coordinate descent
        # x_rcd, f_x_rcd, function_values_rcd, eigen_values_rcd, lip_diag_values_rcd = random_coordinate_descent(
        #     objective, x, lr_rcd, num_iter, sigma, random_keys[exp_i], 
        #     skip_hessian=SKIP_ALL_HESSIAN, 
        #     decay_step=30, decay_rate=0.85
        # )

        # Run block coordinate descent
        x_bcd, f_x_bcd, function_values_bcd, eigen_values_bcd, lip_diag_values_bcd = block_coordinate_descent(
            objective, x, num_iter, sigma, random_keys[exp_i],
            problem_name='max_cut',
            opt_goal='max', 
            opt_method='analytic',
            skip_hessian=SKIP_ALL_HESSIAN, 
            plot_subproblem=True)

        make_dir(f'exp/maxcut/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}')
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

        dump(data_dict, f'exp/maxcut/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl')

if __name__ == "__main__":
    main()