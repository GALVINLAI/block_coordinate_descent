# Gates


## RXGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RXGate

```python
     ┌───────┐
q_0: ┤ Rx(ϴ) ├
     └───────┘
```
Matrix Representation:

$$
R X(\theta)=\exp \left(-i \frac{\theta}{2} X\right)=\left(\begin{array}{cc}
\cos \left(\frac{\theta}{2}\right) & -i \sin \left(\frac{\theta}{2}\right) \\
-i \sin \left(\frac{\theta}{2}\right) & \cos \left(\frac{\theta}{2}\right)
\end{array}\right)
$$

Create new RX gate.

Eigenvalues of -X/2 are: -1/2, 1/2.

## RYGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RYGate

```python
     ┌───────┐
q_0: ┤ Ry(ϴ) ├
     └───────┘
```
Matrix Representation:

$$
R Y(\theta)=\exp \left(-i \frac{\theta}{2} Y\right)=\left(\begin{array}{cc}
\cos \left(\frac{\theta}{2}\right) & -\sin \left(\frac{\theta}{2}\right) \\
\sin \left(\frac{\theta}{2}\right) & \cos \left(\frac{\theta}{2}\right)
\end{array}\right)
$$

Create new RY gate.    

Eigenvalues of -Y/2 are: -1/2, 1/2.

## RZGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZGate

```python
     ┌───────┐
q_0: ┤ Rz(λ) ├
     └───────┘
```
Matrix Representation:

$$
R Z(\lambda)=\exp \left(-i \frac{\lambda}{2} Z\right)=\left(\begin{array}{cc}
e^{-i \frac{\lambda}{2}} & 0 \\
0 & e^{i \frac{\lambda}{2}}
\end{array}\right)
$$

U1Gate This gate is equivalent to U1 up to a phase factor.

$$
U 1(\lambda)=e^{i \lambda / 2} R Z(\lambda)
$$

Create new RZ gate.

Eigenvalues of -Z/2 are: -1/2, 1/2.


## CZGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.CZGate

```python
q_0: ─■─
      │
q_1: ─■─
```

Matrix representation:

$$
C Z q_0, q_1=I \otimes|0\rangle\langle 0|+Z \otimes|1\rangle\langle 1|=\left(\begin{array}{cccc}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1
\end{array}\right)
$$


In the computational basis, this gate flips the phase of the target qubit if the control qubit is in the $|1\rangle$ state.

Create new CZ gate.  



## RXXGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RXXGate

```python
     ┌─────────┐
q_0: ┤1        ├
     │  Rxx(ϴ) │
q_1: ┤0        ├
     └─────────┘
```
Matrix Representation:

$$
R_{X X}(\theta)=\exp \left(-i \frac{\theta}{2} X \otimes X\right)=\left(\begin{array}{cccc}
\cos \left(\frac{\theta}{2}\right) & 0 & 0 & -i \sin \left(\frac{\theta}{2}\right) \\
0 & \cos \left(\frac{\theta}{2}\right) & -i \sin \left(\frac{\theta}{2}\right) & 0 \\
0 & -i \sin \left(\frac{\theta}{2}\right) & \cos \left(\frac{\theta}{2}\right) & 0 \\
-i \sin \left(\frac{\theta}{2}\right) & 0 & 0 & \cos \left(\frac{\theta}{2}\right)
\end{array}\right)
$$


Examples:

$$
\begin{gathered}
R_{X X}(\theta=0)=I \\
R_{X X}(\theta=\pi)=-i X \otimes X \\
R_{X X}\left(\theta=\frac{\pi}{2}\right)=\frac{1}{\sqrt{2}}\left(\begin{array}{cccc}
1 & 0 & 0 & -i \\
0 & 1 & -i & 0 \\
0 & -i & 1 & 0 \\
-i & 0 & 0 & 1
\end{array}\right)
\end{gathered}
$$


Create new RXX gate.

Eigenvalues of $- \frac{1}{2} X \otimes X$ are: -1/2, 1/2.


## RYYGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RYYGate

```python
     ┌─────────┐
q_0: ┤1        ├
     │  Ryy(ϴ) │
q_1: ┤0        ├
     └─────────┘
```
Matrix Representation:

$$
R_{Y Y}(\theta)=\exp \left(-i \frac{\theta}{2} Y \otimes Y\right)=\left(\begin{array}{cccc}
\cos \left(\frac{\theta}{2}\right) & 0 & 0 & i \sin \left(\frac{\theta}{2}\right) \\
0 & \cos \left(\frac{\theta}{2}\right) & -i \sin \left(\frac{\theta}{2}\right) & 0 \\
0 & -i \sin \left(\frac{\theta}{2}\right) & \cos \left(\frac{\theta}{2}\right) & 0 \\
i \sin \left(\frac{\theta}{2}\right) & 0 & 0 & \cos \left(\frac{\theta}{2}\right)
\end{array}\right)
$$


Examples:

$$
\begin{gathered}
R_{Y Y}(\theta=0)=I \\
R_{Y Y}(\theta=\pi)=-i Y \otimes Y \\
R_{Y Y}\left(\theta=\frac{\pi}{2}\right)=\frac{1}{\sqrt{2}}\left(\begin{array}{cccc}
1 & 0 & 0 & i \\
0 & 1 & -i & 0 \\
0 & -i & 1 & 0 \\
i & 0 & 0 & 1
\end{array}\right)
\end{gathered}
$$

Create new RYY gate.

Eigenvalues of $- \frac{1}{2} Y \otimes Y$ are: -1/2, 1/2.

## RZZGate

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZZGate

```python
q_0: ───■────
        │zz(θ)
q_1: ───■────
```
Matrix Representation:

$$
R_{Z Z}(\theta)=\exp \left(-i \frac{\theta}{2} Z \otimes Z\right)=\left(\begin{array}{cccc}
e^{-i \frac{\theta}{2}} & 0 & 0 & 0 \\
0 & e^{i \frac{\theta}{2}} & 0 & 0 \\
0 & 0 & e^{i \frac{\theta}{2}} & 0 \\
0 & 0 & 0 & e^{-i \frac{\theta}{2}}
\end{array}\right)
$$


This is a direct sum of RZ rotations, so this gate is equivalent to a uniformly controlled (multiplexed) RZ gate:

$$
R_{Z Z}(\theta)=\left(\begin{array}{cc}
R Z(\theta) & 0 \\
0 & R Z(-\theta)
\end{array}\right)
$$


Examples:

$$
\begin{gathered}
R_{Z Z}(\theta=0)=I \\
R_{Z Z}(\theta=2 \pi)=-I \\
R_{Z Z}(\theta=\pi)=-i Z \otimes Z \\
R_{Z Z}\left(\theta=\frac{\pi}{2}\right)=\frac{1}{\sqrt{2}}\left(\begin{array}{cccc}
1-i & 0 & 0 & 0 \\
0 & 1+i & 0 & 0 \\
0 & 0 & 1+i & 0 \\
0 & 0 & 0 & 1-i
\end{array}\right)
\end{gathered}
$$


Create new RZZ gate.

Eigenvalues of $- \frac{1}{2} Z \otimes Z$ are: -1/2, 1/2.

## SdgGate 

https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.SdgGate

```python
     ┌─────┐
q_0: ┤ Sdg ├
     └─────┘
```
Matrix Representation:
$$
    Sdg = \begin{pmatrix}
    1 & 0 \\
    0 & -i
    \end{pmatrix}
$$
In quantum computing, `SdgGate` typically refers to the **S† (S dagger) gate**, which is the Hermitian conjugate (or inverse) of the **S gate**. Specifically, the S and S† gates are quantum gates associated with **phase rotations**, representing positive and negative phase rotations, respectively.

### 1. S Gate
The S gate (also known as the **Phase Gate**) is a 2×2 matrix that acts on a single qubit. The effect of the S gate on a qubit state is to rotate the phase of the qubit. Its matrix representation is:
$$
    S = \begin{pmatrix}
    1 & 0 \\
    0 & i
    \end{pmatrix}
$$
Its action is to multiply the $\left| 1 \right\rangle$ state of the qubit by a phase factor $i$, i.e.,
$$
    S \left| 0 \right\rangle = \left| 0 \right\rangle, \quad S \left| 1 \right\rangle = i \left| 1 \right\rangle
$$
### 2. S dagger Gate
The `S†` gate is the Hermitian conjugate of the S gate, i.e., the transpose conjugate of the matrix. Its matrix representation is:
$$
    S^\dagger = \begin{pmatrix}
    1 & 0 \\
    0 & -i
    \end{pmatrix}
$$
The effect of the S† gate is to apply a negative phase factor $-i$ to the qubit, i.e.,
$$
    S^\dagger \left| 0 \right\rangle = \left| 0 \right\rangle, \quad S^\dagger \left| 1 \right\rangle = -i \left| 1 \right\rangle
$$


## 检查模拟器结果
- 如何在模拟器上测量

Measure qubits https://docs.quantum.ibm.com/guides/measure-qubits

How to measure in $\sigma_x$ basis in Qiskit?
https://quantumcomputing.stackexchange.com/questions/34860/how-to-measure-in-sigma-x-basis-in-qiskit

How to measure in another basis
https://quantumcomputing.stackexchange.com/questions/13605/how-to-measure-in-another-basis

![alt text](Bil8h.png)

- 模拟器的观测值是否是真实值的正态分布
More generally, for an arbitrary number of qubits, when you want to measure in a given basis, it will always be possible to write this basis as $(U|i\rangle)_i$. If you apply $U^{\dagger}$ to $|\psi\rangle$ and measure in the computational basis, you will measure $|i\rangle$ with probability:
$$
    \left.\left|\langle i| U^{\dagger}\right| \psi\right\rangle\left.\right|^ 2
$$
Which is exactly the probability to measure $U|i\rangle$ if we measure in the $(U|i\rangle)_i$ basis. You then simply need to apply $U$ to simulate a measurement in this basis using one in the computational basis.


Measure in Pauli bases https://docs.quantum.ibm.com/guides/specify-observables-pauli 

A measurement projects the qubit state to the computational basis $\{|0\rangle,|1\rangle\}$. This implies that you can only measure observables that are diagonal in this basis, such as Paulis consisting only of $I$ and $Z$ terms. Measuring arbitrary Pauli terms therefore requires a change of basis to diagonalize them. To do this, perform the following transformations,

$$
\begin{aligned}
& X \rightarrow Z=H X H \\
& Y \rightarrow Z=H S^{\dagger} Y S H
\end{aligned}
$$

where $H$ is the Hadamard gate and $S=\sqrt{Z}$ is sometimes referred to as the phase gate. If you are using an Estimator to compute expectation values, the basis transformations are automatically performed.