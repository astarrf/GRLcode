import numpy as np
import math as mt
from scipy.integrate import solve_ivp


class DMatrix():
    def __init__(self, N: int, n_ord: int):
        self.N = N
        self.n_ord = n_ord

    def get_a_n(self):
        return self.a_n_matrix

    def get_n_ord(self):
        return self.n_ord

    def set_a_n_m_matrix(self, m: int):
        diagonal_value = np.zeros(self.N-m)
        sub_diagonal_value = np.zeros(self.N-m-self.n_ord)
        for i in range(self.N-m):
            P_i_n = mt.perm(i, self.n_ord)
            P_i_n_plus_m = mt.perm(i+m, self.n_ord)
            diagonal_value[i] = -(P_i_n+P_i_n_plus_m)/2
        for i in range(self.N-m-self.n_ord):
            P_i_plus_n_n = mt.perm(i+self.n_ord, self.n_ord)
            P_i_plus_n_plus_m_n = mt.perm(i+self.n_ord+m, self.n_ord)
            sub_diagonal_value[i] = mt.sqrt(P_i_plus_n_n*P_i_plus_n_plus_m_n)
        self.a_n_matrix = np.zeros((self.N-m, self.N-m))
        # print(diagonal_value)
        # print(sub_diagonal_value)
        np.fill_diagonal(self.a_n_matrix, diagonal_value)
        np.fill_diagonal(self.a_n_matrix[:, self.n_ord:], sub_diagonal_value)


class LMatrix():
    def __init__(self, N: int):
        self.N = N

    def get_L(self):
        return self.L_matrix

    def set_L_list(self, L_list: list):
        tmp_L_list = [0]*(self.N)
        for l in L_list:
            idx = int(l[0]-1)
            if idx <= self.N-1:
                tmp_L_list[idx] = l[1]
        self.L_list = tmp_L_list
        self.Lambda = np.sum([l**2 for l in self.L_list])

    def set_L_matrix(self, m: int, L_list: list):
        # L_list can be a list of integers, representing the position of the L operator
        # or a list of list, representing the position of the L operator and the coefficient of the L operator
        self.set_L_list(L_list)
        self.L_matrix = np.zeros((self.N-m, self.N-m))
        L_list = self.L_list
        for i in range(self.N-m-1):
            self.L_matrix[i+1, i+1] = -(L_list[i+1]**2+L_list[i+m+1]**2) / 2
            self.L_matrix[i+1, i] = L_list[i]*L_list[i+m]
        self.L_matrix /= self.Lambda


class MasterEqu():
    def __init__(self, N: int, en1: list, en0: list, N_ord: int, Dis_list: list, lam: float, L_list: list):
        self.N = N
        self.en1 = en1[:, 0]
        self.en0 = en0[:, 0]
        self.en1_coef = en1[:, 1]
        self.en0_coef = en0[:, 1]
        self.N_ord = N_ord
        self.L_list = L_list
        self.Dis_list = Dis_list
        self.lam = lam

    def get_m_list(self):
        m_list = [0]
        for e1 in self.en1:
            for e0 in self.en0:
                num = int(abs(e1-e0))
                if num not in m_list:
                    m_list.append(num)
        m_list.sort()
        return m_list

    def get_master_equ(self):
        m_list = self.get_m_list()  # 求出所有需要求解的对角线编号
        matx_list = [np.zeros((self.N-m, self.N-m), dtype=complex)
                     for m in m_list]  # 初始化所有对角线方程的系数矩阵
        for n_ord in range(1, self.N_ord+1):
            a = DMatrix(self.N, n_ord)
            for i, m in enumerate(m_list):
                a.set_a_n_m_matrix(m)
                a_m_diag = a.get_a_n()
                matx_list[i] += (a_m_diag*self.Dis_list[n_ord-1])
        l = LMatrix(self.N)
        for i, m in enumerate(m_list):
            l.set_L_matrix(m, self.L_list)
            l_diag = l.get_L()
            matx_list[i] += l_diag*self.lam
        return matx_list

    def get_eigen_mtx(self):
        self.m_list = self.get_m_list()
        self.matx_list = self.get_master_equ()
        encode_list = (self.en0+self.en1)
        encode_list.sort()
        self.Encode_list = np.array(encode_list)

    def get_basic_states(self):
        N = self.N
        basis_0 = np.zeros((N), dtype=complex)
        basis_0[self.en0] = self.en0_coef
        basis_1 = np.zeros((N), dtype=complex)
        basis_1[self.en1] = self.en1_coef
        # get the pauli matrices of the basis
        pauli_x = np.outer(basis_0.T, basis_1)+np.outer(basis_1.T, basis_0)
        pauli_y = 1j*(np.outer(basis_0.T, basis_1) -
                      np.outer(basis_1.T, basis_0))
        pauli_z = np.outer(basis_0.T, basis_0)-np.outer(basis_1.T, basis_1)
        pauli_0 = np.outer(basis_0.T, basis_0)+np.outer(basis_1.T, basis_1)
        pauli_x.dtype = complex
        pauli_y.dtype = complex
        pauli_z.dtype = complex
        pauli_0.dtype = complex
        self.rhos = [(pauli_0+pauli_x)/2, (pauli_0-pauli_x)/2,
                     (pauli_0+pauli_y)/2, (pauli_0-pauli_y)/2,
                     (pauli_0+pauli_z)/2, (pauli_0-pauli_z)/2]

    def get_eigen_fid(self, glob_mtx, t_eval):
        N = self.N

        if glob_mtx.shape != (N, N):
            raise ValueError("The shape of the input matrix is not correct.")
        if glob_mtx.dtype != complex:
            raise ValueError("The input matrix is not complex.")

        m_list = self.m_list
        glob_res_list = [np.zeros((N, N), dtype=complex)
                         for _ in t_eval]

        for i in range(len(m_list)):
            coef = np.diagonal(glob_mtx, offset=-m_list[i])
            def F(t, y): return self.matx_list[i] @ y
            sol = solve_ivp(F, [0, t_eval[-1]], coef,
                            t_eval=t_eval, vectorized=True)
            idx = m_list[i]
            idxN = self.N-idx
            for j in range(len(t_eval)):
                s = sol.y[:, j]
                tmp_res = np.zeros((N, N), dtype=complex)
                if idx == 0:
                    np.fill_diagonal(tmp_res, s)
                else:
                    np.fill_diagonal(tmp_res[idx:, :idxN], s)
                    np.fill_diagonal(tmp_res[:idxN, idx:], np.conj(s))
                glob_res_list[j] += tmp_res

        # fidelity of all time
        fid_list = [np.abs(np.trace(glob_mtx @ glob_res_list[i]))
                    for i in range(len(t_eval))]
        return fid_list, glob_res_list[-1]

    def get_fid_basic_matrices(self, rhos, t_eval):
        fid_list = []
        current_rhos = []
        for rho0 in rhos:
            fid, current_rho = self.get_eigen_fid(rho0, t_eval)
            fid_list.append(fid)
            current_rhos.append(current_rho)
        return fid_list, current_rhos
