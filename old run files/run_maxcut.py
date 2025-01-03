import argparse
import os, shutil

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit

from algo.utils import dump, make_dir, hamiltonian_to_matrix
from algo.utils_qiskit import find_pauli_indices, process_hamiltonian_Zs
from algo.check_utils import interp_matrix, check_is_trigometric

from algo.oicd_qiskit import oicd
from algo.gd_qiskit import gd
from algo.rcd_qiskit import rcd

# Set up configurations
# config.update("jax_enable_x64", True)
# np.random.seed(6)
# adding the configuration here


def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # # Add the system size argument
    # parser.add_argument('--N', type=int, default=4, help='System size')

    # # Add the problem dimension argument
    # parser.add_argument('--dim', type=int, default=20, 
    #                     help='The dimension of the problem')
    
    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=10, 
                        help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.05, 
                        help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.1, 
                        help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=300, 
                    help='The number of iterations for the optimization algorithm')
    
    parser.add_argument('--n_shot', type=int, default=1000, 
                    help='The number of shots for each evluation')
        
    return parser

args = create_parser().parse_args()

print("Run the HEA algorithm for the maxcut model")
# print(f"System size: {args.N}")
# print(f"Problem dimension: {args.dim}")
print(f"Repeat count: {args.repeat}")
print(f"Gradient descent learning rate: {args.lr_gd}")
print(f"Random coordinate descent learning rate: {args.lr_rcd}")
print(f"Number of iterations: {args.num_iter}")

# num_q = args.N
# dim = args.dim
repeat = args.repeat
lr_gd = args.lr_gd
lr_rcd = args.lr_rcd
num_iter = args.num_iter
n_shot = args.n_shot

######################## max-cut problem setup ########################

# Hamiltonian string
ham_str = '0.5 - 3 * z0  + 0.5 * z1 * z0 + 0.5 * z2 * z0 + 0.5 * z2 * z1 + 0.5 * z3 * z0 + 0.5 * z3 * z2'

# Number of qubits
num_q = 4

# Call the function to process the Hamiltonian string
all_lists, all_coeffs = process_hamiltonian_Zs(ham_str, num_q, max_stars=2)

# # Output the results for each star category
# for i, (term_positions, term_coeffs) in enumerate(zip(all_lists, all_coeffs)):
#     print(f"Terms with {i} star(s):")
#     print("Coefficients:", term_coeffs)
#     print("Positions:", term_positions)

# print('')

List_0, List_1, List_2 = all_lists
List = List_0 + List_1 + List_2

coeff_0, coeff_1, coeff_2 = all_coeffs
coeff = coeff_0 + coeff_1 + coeff_2

H = SparsePauliOp(List, coeff)
Hmat = Operator(H)
Hmat = Hmat.data

# # Check that it is the same as the matrix in the jax problem
# H2 = hamiltonian_to_matrix(ham_str)
# print('difference between two matrices', np.linalg.norm(H2-Hmat.data))

# Compute the eigenvalues and right eigenvectors of a square array.
e, v = LA.eig(Hmat) 
min_index = np.argmin(e)
v_min = v[:,min_index] # ground state (eigenvector)
ground_e = np.min(e) # ground state energy
print('ground energy',ground_e)
print('ground state ',v_min)


######################## circuit construction ########################

# Define the number of layers in the quantum circuit
layer = 5  

# Calculate the number of parameters 
# This needs to be determined based on the circuit
num_p = num_q * layer 

# Hardware-Efficient Ansatz
def circuit_HEA(weights):
    # A QuantumCircuit with 4 qubits and 3 classical bits
    circ = QuantumCircuit(num_q, num_q)  
    
    for i in range(layer):
        # initial state is not |0⟩^⊗n
        for j in range(num_q): 
            circ.ry(np.pi/4, j) # Apply RY gate to each qubit, the angle is given by elements in the weights array
        for j in range(num_q-1):  
            circ.cz(j, j+1) # Apply CZ gate to each pair of adjacent qubits

    for i in range(layer):  # Iterate over each layer
        for j in range(num_q): 
            circ.ry(weights[num_q*i+j], j)  # Apply RY gate to each qubit, the angle is given by elements in the weights array
        for j in range(num_q-1):  
            circ.cz(j, j+1) # Apply CZ gate to each pair of adjacent qubits
    
    return circ # Return the constructed quantum circuit

# Create a vector of parameters (parameters of the quantum circuit)
weights = ParameterVector("w", num_p)  
qc = circuit_HEA(weights)
qc.draw("mpl")


weights_dict = {}

omegas = [1]
interp_nodes = np.linspace(0,2*np.pi,2*len(omegas)+1,endpoint=False)
# interp_nodes = np.random.uniform(0, 2*np.pi, size=2*len(omegas)+1)
inverse_interp_matrix = np.linalg.inv(interp_matrix(interp_nodes, omegas))

for i in range(num_p):
    weights_dict[f'weights_{i}'] = {
        'omegas': omegas,
        'interp_nodes': interp_nodes,
        'inverse_interp_matrix': inverse_interp_matrix,
    }

print(weights_dict['weights_0'])
print(f'true ground state energy:',ground_e)


######################## loss and fidelity function construction ########################

simulator = AerSimulator()

Z_indices = []
ZZ_indices = []

for pauli_str in List_1:
    # print(pauli_str)
    _, _, Z_index = find_pauli_indices(pauli_str)
    Z_indices.append(Z_index)

for pauli_str in List_2:
    # print(pauli_str)
    _, _, ZZ_index = find_pauli_indices(pauli_str)
    ZZ_indices.append(ZZ_index)

def estimate_loss(WEIGHTS, SHOTS):

    estimate_1 = 0 
    estimate_2 = 0

    qc = circuit_HEA(WEIGHTS)
    qc = transpile(qc, simulator)
    ind = list(range(num_q))
    rind = ind
    rind.reverse()
    qc.measure(ind, rind)
    result = simulator.run(qc, shots = SHOTS, memory=True).result()
    c = result.get_memory(qc)

    for i in range(SHOTS):
        
        c_i = c[i]

        # List_1 = ['ZIII'] Terms with 1 star
        for j, index in enumerate(Z_indices): 
            if c_i[num_q-1-index[0]] == '0':
                estimate_1 += 1*coeff_1[j]
            else:
                estimate_1 += -1*coeff_1[j]
        
        # List_2 = ['ZZII', 'ZIZI', 'IZZI', 'ZIIZ', 'IIZZ'] Terms with 2 stars
        for j, index in enumerate(ZZ_indices): 
            if c_i[num_q-1-index[0]] == c_i[num_q-1-index[1]]:
                estimate_2 += 1*coeff_2[j]
            else:
                estimate_2 += -1*coeff_2[j]
        

    estimate = estimate_1 + estimate_2
    estimate = estimate/SHOTS + coeff_0[0] # coeff_0[0] for 'IIII'

    return estimate


def expectation_loss(WEIGHTS):
    qc = circuit_HEA(WEIGHTS)
    qc.save_statevector()
    qc = transpile(qc, simulator)
    result = simulator.run(qc).result()
    state_vector = result.get_statevector(qc)
    psi = np.asarray(state_vector)
    # ==========================================================================
    Hpsi = Hmat.dot(psi)
    expectation = np.inner(np.conjugate(psi),Hpsi)
    return np.real(expectation)

def fidelity(WEIGHTS):
    qc = circuit_HEA(WEIGHTS)
    qc.save_statevector()
    qc = transpile(qc, simulator)
    result = simulator.run(qc).result()
    state_vector = result.get_statevector(qc)
    psi = np.asarray(state_vector)
    # ==========================================================================
    return np.absolute(np.vdot(psi,v_min))**2

def std(WEIGHTS):
    qc = circuit_HEA(WEIGHTS)
    qc.save_statevector()
    qc = transpile(qc, simulator)
    result = simulator.run(qc).result()
    state_vector = result.get_statevector(qc)
    psi = np.asarray(state_vector)
    # ==========================================================================
    Hmat_sqaured =  Hmat @ Hmat
    Hmat_sqauredpsi = Hmat_sqaured.dot(psi)
    var = np.inner(np.conjugate(psi),Hmat_sqauredpsi) - expectation_loss(WEIGHTS)**2
    return np.sqrt(np.real(var))


def energy_ratio(WEIGHTS):
    return np.abs(expectation_loss(WEIGHTS)/ ground_e)

######################## Solver setup ########################

def main():
    # Check if the folder exists and delete it if it does.
    # Note that we delete everything inside the folder
    make_dir('exp/maxcut')

    dir_path = f'exp/maxcut/'
    
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed existing directory: {dir_path}")

    for exp_i in range(repeat):

        print('='*100)

        print(f'Experiment # {exp_i} begins.')

        # Define the initial value for x
        # Ensure that the initial point of each experiment exp_i is random
        initial_weights = np.random.uniform(0, 2*np.pi, size=num_p)

        # Initialize data_dict
        data_dict = {}

        ############################################################
        # Run gradient descent
        final_weights_gd, best_expected_record_value_gd, fidelity_record_value_gd, func_gd = gd(
            estimate_loss,
            expectation_loss,
            fidelity,
            ground_e,
            n_shot, weights_dict, initial_weights, num_iter,
            learning_rate=lr_gd,
            exact_mode=False,
            plot_flag=False,
        )

        data_dict.update({
            'x_gd': final_weights_gd,
            'best_expected_gd': best_expected_record_value_gd / ground_e,
            'fidelity_gd': fidelity_record_value_gd,
            'func_gd': func_gd
        })
        
        ############################################################
        # Run random coordinate descent
        final_weights_rcd, best_expected_record_value_rcd, fidelity_record_value_rcd, func_rcd= rcd(
            estimate_loss,
            expectation_loss,
            fidelity,
            ground_e,
            n_shot, weights_dict, initial_weights, num_iter,
            learning_rate=lr_rcd,
            cyclic_mode=False,
            exact_mode=False,
            plot_flag=False,
        )

        data_dict.update({
            'x_rcd': final_weights_rcd,
            'best_expected_rcd': best_expected_record_value_rcd / ground_e,
            'fidelity_rcd': fidelity_record_value_rcd,
            'func_rcd': func_rcd
        })

        ############################################################
        # Run OICD
        final_weights_oicd, best_expected_record_value_oicd, fidelity_record_value_oicd, func_oicd = oicd(
            estimate_loss,
            expectation_loss,
            fidelity,
            ground_e,
            n_shot, weights_dict, initial_weights, num_iter,
            subproblem_method='CG',  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            subproblem_iter=None,
            cyclic_mode=False,  # False is very fast.
            use_local_solvers_flag=False,
            use_pratical_interp_flag=True,
            exact_mode=False,
            plot_flag=False,
        )

        data_dict.update({
            'x_oicd': final_weights_oicd,
            'best_expected_oicd': best_expected_record_value_oicd/ ground_e,
            'fidelity_oicd': fidelity_record_value_oicd,
            'func_oicd': func_oicd
        })

        ############################################################
        make_dir(f'exp/maxcut/exp_{exp_i}')
        dump(data_dict, f'exp/maxcut/exp_{exp_i}/data_dict.pkl')

if __name__ == "__main__":
    main()
