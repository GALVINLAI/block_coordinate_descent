import argparse
import os, shutil
import matplotlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from utils import dump, make_dir, hamiltonian_to_matrix

from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from algo.bcd_dev import block_coordinate_descent
from algo.oicd import oicd

# Set up configurations
# matplotlib.use("Agg")  # Set the matplotlib backend to 'Agg'
# np.random.seed(42)

def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add the system size argument
    parser.add_argument('--N', type=int, default=4, help='System size')

    # Add the problem dimension argument
    parser.add_argument('--dim', type=int, default=20, help='The dimension of the problem')
    
    # Add the sigma argument
    parser.add_argument('--sigma', type=float, default=0.1, help='The sigma for the Gaussian noise of the gradient. Note that this is multiplied by the gradient')
    
    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=10, help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.0001, help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.001, help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=1000, help='The number of iterations for the optimization algorithm')

    return parser

args = create_parser().parse_args()
print("Run the QAOA algorithm for the TSP model")
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

ham_str = '600303.0 -100069.5 * z0 -100055.5 * z4 + 12.0 * z4 * z0 -100069.5 * z1 -100055.5 * z5 + 12.0 * z5 * z1 -100069.5 * z2 -100055.5 * z3 + 12.0 * z3 * z2 -100077.0 * z7 + 22.75 * z7 * z0 -100077.0 * z8 + 22.75 * z8 * z1 -100077.0 * z6 + 22.75 * z6 * z2 + 12.0 * z3 * z1 + 12.0 * z4 * z2 + 12.0 * z5 * z0 + 15.75 * z7 * z3 + 15.75 * z8 * z4 + 15.75 * z6 * z5 + 22.75 * z6 * z1 + 22.75 * z7 * z2 + 22.75 * z8 * z0 + 15.75 * z6 * z4 + 15.75 * z7 * z5 + 15.75 * z8 * z3 + 50000.0 * z3 * z0 + 50000.0 * z6 * z0 + 50000.0 * z6 * z3 + 50000.0 * z4 * z1 + 50000.0 * z7 * z1 + 50000.0 * z7 * z4 + 50000.0 * z5 * z2 + 50000.0 * z8 * z2 + 50000.0 * z8 * z5 + 50000.0 * z1 * z0 + 50000.0 * z2 * z0 + 50000.0 * z2 * z1 + 50000.0 * z4 * z3 + 50000.0 * z5 * z3 + 50000.0 * z5 * z4 + 50000.0 * z7 * z6 + 50000.0 * z8 * z6 + 50000.0 * z8 * z7'
H = hamiltonian_to_matrix(ham_str)

print(jnp.linalg.eigh(H)[0][0])
print(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0]))
print(bin(int(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0])[0][0]))[2:])

ans = list(str(bin(int(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0])[0][0]))[2:]))[::-1]
ans = jnp.array([int(i) for i in ans] + [0] * (9 - len(ans)))
print(ans)

def get_tsp_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray): Binary string as numpy array.

    Returns:
        list[int]: Sequence of cities to traverse.
            The i-th item in the list corresponds to the city which is visited in the i-th step.
            The list for an infeasible answer e.g. [[0,1],1,] can be interpreted as
            visiting [city0 and city1] as the first city, then visit city1 as the second city,
            then visit nowhere as the third city).
    """
    n = int(np.sqrt(len(x)))
    z = []
    for p__ in range(n):
        p_th_step = []
        for i in range(n):
            if x[i * n + p__] >= 0.999:
                p_th_step.append(i)
        if len(p_th_step) == 1:
            z.extend(p_th_step)
        else:
            z.append(p_th_step)
    return z

print(get_tsp_solution(ans))

E_gs = jnp.linalg.eigh(H)[0][0]

def ry(theta):
    """Create a rotation matrix for a rotation about the y-axis."""
    return jnp.array([[jnp.cos(theta / 2), -jnp.sin(theta / 2)],
                      [jnp.sin(theta / 2), jnp.cos(theta / 2)]])

def cz():
    """Create a controlled-Z gate."""
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])

def _apply_gate(state, gate, targets, n_qubits):
    """Apply a gate to a state."""
    operators = [jnp.eye(2)] * n_qubits
    
    if len(targets) == 1:
        operators[targets[0]] = gate
    else:
        operators[targets[0]: targets[1] + 1] = [gate]
    
    gate = operators[0]
    for operator in operators[1:]:
        gate = jnp.kron(gate, operator)
    return jnp.dot(gate, state)

def apply_gates(state, n_qubits, reps, ry_angles):
    """Apply the sequence of RY and CZ gates."""
    for rep in range(reps):
        for i in range(n_qubits):
            state = _apply_gate(state, ry(ry_angles[rep * n_qubits + i]), [i], n_qubits)
        for i in range(n_qubits - 1):
            state = _apply_gate(state, cz(), [i, i + 1], n_qubits)
    return state

@jax.jit
def get_energy_ratio(psi):
    return -jnp.vdot(psi, jnp.dot(H, psi)) / E_gs

# Define the number of qubits
n_qubits = 3 ** 2

# Define the number of repetitions
reps = 10

dim = n_qubits * reps

# Define the rotation angles for the RY gates
ry_angles = [jnp.pi / 4] * dim  # Using pi/4 for each qubit as an example

# Create the initial state
state = jnp.eye(2 ** n_qubits)[:, 0]

# Apply the gates
state = apply_gates(state, n_qubits, reps, ry_angles)

@jax.jit
def objective(params):
    psi = apply_gates(state, n_qubits, reps, params)
    return get_energy_ratio(psi)

# The final state is now stored in the 'state' variable
print(objective(ry_angles))

# Set up configurations
# config.update("jax_enable_x64", True)
# np.random.seed(6)

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
    parser.add_argument('--repeat', type=int, default=1, 
                        help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=1.0, 
                        help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=1.0, 
                        help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=500, 
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
    angle_range = np.linspace(-4 * np.pi, 4 * np.pi, 4000)  # Angle range from -4π to 4π

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

    print("极大值点:", angle_range[peaks])

    # If at least two peaks are found, calculate the distances between adjacent peaks
    if len(peaks) > 1:
        peak_distances = np.diff(angle_range[peaks])  # Calculate the distances between adjacent peaks
        print("相邻的极大值点之间的距离:", peak_distances)
    else:
        print("没有找到足够的极大值点")


# # Example usage:
# # Assume `objective` is your function and `ry_angles` is your parameter list
# plot_single_objective(objective, ry_angles, 2, plot_flag=True)  # To plot the graph
# plot_single_objective(objective, ry_angles, 2, plot_flag=False)  # To only calculate and print peaks without plotting

# Loop through all positions in the params array
for j in range(len(ry_angles)):
    print(f"Processing angle at index {j} (params[{j}])")
    plot_single_objective(objective, ry_angles, j, plot_flag=False)


