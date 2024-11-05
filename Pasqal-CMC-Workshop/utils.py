import numpy as np
import matplotlib.pyplot as plt
import qutip

import pulser
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.waveforms import RampWaveform
from pulser.devices import AnalogDevice

def occupation(j, N):
    up = qutip.basis(2, 0)
    prod = [qutip.qeye(2) for _ in range(N)]
    prod[j] = up * up.dag()
    return qutip.tensor(prod)

def get_corr_pairs(k, l, register, R_interatomic):
    corr_pairs = []
    for i, qi in enumerate(register.qubits):
        for j, qj in enumerate(register.qubits):
            r_ij = register.qubits[qi] - register.qubits[qj]
            distance = np.linalg.norm(r_ij - R_interatomic * np.array([k, l]))
            if distance < 1:
                corr_pairs.append([i, j])
    return corr_pairs

def get_corr_function(k, l, reg, R_interatomic, state):
    N_qubits = len(reg.qubits)
    corr_pairs = get_corr_pairs(k, l, reg, R_interatomic)

    operators = [occupation(j, N_qubits) for j in range(N_qubits)]
    covariance = 0
    for qi, qj in corr_pairs:
        covariance += qutip.expect(operators[qi] * operators[qj], state)
        covariance -= qutip.expect(operators[qi], state) * qutip.expect(
            operators[qj], state
        )
    return covariance / len(corr_pairs)
    
def correl_total(N_side, R_interatomic, Omega_max, reg, state):
    U = Omega_max / 2.0
    
    occup_list = [occupation(j, N_side * N_side) for j in range(N_side * N_side)]
    
    def get_full_corr_function(reg, state):
        N_qubits = len(reg.qubits)
    
        correlation_function = {}
        N_side = int(np.sqrt(N_qubits))
        for k in range(-N_side + 1, N_side):
            for l in range(-N_side + 1, N_side):
                correlation_function[(k, l)] = get_corr_function(
                    k, l, reg, R_interatomic, state
                )
        return correlation_function
    return get_full_corr_function(reg, state)