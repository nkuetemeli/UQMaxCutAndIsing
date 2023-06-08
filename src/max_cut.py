import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
from src.utils import *

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
from qiskit.transpiler.passes import RemoveBarriers
import time


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


font_size = 10
params = {'axes.labelsize': font_size,
          'axes.titlesize': font_size, }
plt.rcParams.update(params)


class MaxCut():
    def __init__(self,
                 n=10,
                 p=(1., 1.),
                 seed=10,
                 weight_bounds=(1, 10),
                 G=None,
                 shots=1*1024,
                 coef_K=2/np.pi,
                 thetas_init=None,
                 brute_force=False,
                 exact_costs=True,
                 entanglement=None,
                 decompose=True,
                 ):

        self.seed = seed
        self.weight_bounds = weight_bounds
        G = self.get_graph(n, p, weight_bounds) if G is None else G
        adjacency = np.triu(nx.adjacency_matrix(G).toarray())
        if entanglement == 'bell':
            thetas_init = np.zeros(n-1,) if thetas_init is None else thetas_init
        else:
            thetas_init = np.zeros(n,) if thetas_init is None else thetas_init

        self.G = G
        self.n = n
        self.coef_K = coef_K
        self.K = coef_K * np.sum(np.abs(adjacency))
        self.adjacency = adjacency
        self.shots = shots
        self.thetas_init = thetas_init
        self.thetas_opt = self.thetas_init
        self.brute_force = brute_force
        self.exact_costs = exact_costs
        self.entanglement = entanglement
        self.decompose = decompose
        self.history = {'loss': [], 'norm_grad': [], 'ratio': [], 'index': []}
        self.time = 0

        # Brute force the problem to compare the results
        if self.brute_force and self.entanglement is None:
            thetas = lambda i: np.pi * np.array([int(x) for x in np.binary_repr(i, width=self.n)])
            _mcc = self._mc_circuit()
            U = Operator(_mcc.reverse_bits()).data
            U = np.arcsin(U) if self.exact_costs else U
            diagU = np.round(np.real(self.K * np.diag(U)), 2)
            self.costs = np.array([self.classical_cost(np.binary_repr(i, width=self.n)) for i in range(2**self.n)])
            self.acosts = diagU[:2**self.n]
            self.qcosts = np.array([self.eval(thetas(i)) for i in range(2**self.n)])
            self.qcosts = np.array([self.K * np.arcsin(self.eval(thetas(i)) / self.K) for i in range(2**self.n)])
            print('True costs\n', list(np.round(self.costs, 0)))
            print('Approximated costs\n', list(np.round(self.acosts, 0).astype(np.int32)))
            print('Quantum costs\n', list(np.round(self.qcosts, 0).astype(np.int32)))
        else:
            self.costs = np.array([self.classical_cost(np.binary_repr(i, width=self.n)) for i in range(2**self.n)])
            self.qcosts = None

        index = np.nonzero(self.costs == self.costs.min())[0]
        index = [np.binary_repr(x, width=self.n) for x in index]
        expected_cut = "\n or".join([f' {self.cut(index[i])[0]} and {self.cut(index[i])[1]}' for i in range(len(index))])
        self.expected_cut = f'\nExpected Cut = {expected_cut}'


    def cut(self, index):
        result = (list(np.where(np.array([int(x) for x in index]) == 0)[0]),
                  list(np.where(np.array([int(x) for x in index]) == 1)[0]))
        return result

    def get_graph(self, n=10, p=(1., 1.), weight_bounds=(1, 10)):
        np.random.seed(self.seed)
        G = nx.gnp_random_graph(n, p[0], seed=self.seed)
        choosen_edges = np.random.choice(n, int(p[1]*n), replace=False)
        np.random.seed(self.seed)
        for u, v in G.edges:
            G.add_edge(u, v, weight=np.random.randint(*weight_bounds))
        for u in choosen_edges:
            G.add_edge(u, u, weight=np.random.randint(*weight_bounds))
        return G

    def draw_graph(self, colors=None, title='', save=False):
        colors = self.n*['lightblue'] if colors is None else colors
        labels = nx.get_edge_attributes(self.G, 'weight')
        pos = {u: (np.cos(2*np.pi*u/self.n), np.sin(2*np.pi*u/self.n)) for u in self.G.nodes}
        plt.figure()
        plt.title(title)
        nx.draw(self.G, pos, with_labels=True, node_color=colors)
        _ = nx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=labels, font_size=font_size)
        if save:
            if title == '':
                save_fig(plt.gcf(), path='results/images/graph_init.pdf')
            else:
                save_fig(plt.gcf(), path='results/images/graph_opt.pdf')
        plt.show()
        return

    def draw_circuit(self, thetas=None):
        thetas = self.thetas_init if thetas is None else thetas
        circuit = self.mc_circuit(thetas)
        circuit.draw(output='mpl')
        plt.show()
        return

    def edge_circuit(self, circuit, q0, q1, q2, C, K):

        if q0 == q1:
            circuit.cx(q1, q2)
            circuit.ry(-2 * C / K, [q2])
            circuit.cx(q0, q2)
            circuit.barrier([q0, q2])
        else:
            circuit.cx(q1, q2)
            circuit.cx(q0, q2)
            circuit.ry(-2 * C / K, [q2])
            circuit.cx(q0, q2)
            circuit.cx(q1, q2)
            circuit.barrier([q0, q1, q2])
        return circuit

    def entanglement_circuit(self):
        circuit = QuantumCircuit(self.n, name=' Entanglement ')

        if self.entanglement == 'bell':
            for i in range(self.n-1, 0, -1):
                circuit.cx(0, i)
            circuit.h(0)
            for i in range(1, self.n):
                circuit.cx(0, i)

        elif self.entanglement == 'linear':
            for i in range(1, self.n):
                circuit.cx(i-1, i)

        elif self.entanglement == 'full':
            for i in range(0, self.n-1):
                for j in range(i+1, self.n):
                    circuit.cx(i, j)

        elif self.entanglement == 'circular':
            for i in range(self.n):
                circuit.cx(i, (i+1) % self.n)

        else:
            assert False, "Parameter 'entanglement' should be something from [None, 'bell', 'full', 'linear', 'circular']"

        return circuit

    def rotation_circuit(self, thetas):
        circuit = QuantumCircuit(self.n, name=' Rotations ')
        for i, theta in enumerate(thetas):
            if self.entanglement == 'bell':
                circuit.ry(theta, i+1)
            else:
                circuit.ry(theta, i)
        return circuit


    def _mc_circuit(self):
        qc = QuantumRegister(1, name='c')
        qq = QuantumRegister(self.n, name='q')

        circuit = QuantumCircuit(qc, qq, name=' Maximum Cut ')
        circuit.x(qc[0])

        for u, v in list(self.G.edges):
            circuit = self.edge_circuit(circuit, q0=qq[u], q1=qq[v], q2=qc[0], C=self.G[u][v]['weight'], K=self.K)
        return circuit

    def mc_circuit(self, thetas):
        n = self.n

        _mcc = self._mc_circuit()

        qa = QuantumRegister(1, name='a')
        qc = QuantumRegister(1, name='c')
        qq = QuantumRegister(n, name='q')
        qm = ClassicalRegister(1, name='m')

        circuit = QuantumCircuit(qa, qc, qq, qm)

        circuit.append(self.rotation_circuit(thetas).to_gate(), list(range(2, self.n + 2)))

        if self.entanglement is not None:
            circuit.append(self.entanglement_circuit().to_gate(), list(range(2, self.n + 2)))

        if self.decompose:
            circuit = circuit.decompose(' Rotations ')
            circuit = circuit.decompose(' Entanglement ')

        circuit.barrier()

        circuit.h(qa[0])
        circuit.append(RemoveBarriers()(_mcc).to_gate().control(1), [qa, qc] + [qq[x] for x in range(n)])
        circuit.h(qa[0])

        circuit.barrier()

        circuit.measure(qa, qm[0])

        return circuit

    def eval(self, thetas):
        circuit = self.mc_circuit(thetas)
        counts = self.run(circuit)
        val = self.quantum_cost(counts)
        return val

    def grad(self, thetas, iter, gradient_method='spsa'):
        grad = Grad(f=self.eval, x=thetas, iter=iter, gradient_method=gradient_method)
        return grad()

    def eval_ratio(self, counts):
        max_cost = np.max(self.costs)
        min_cost = np.min(self.costs)
        max_index = max(counts, key=counts.get)
        ratio = (self.classical_expectation(counts) - max_cost) / (min_cost - max_cost)
        index = (min_cost == self.classical_cost(max_index))
        return ratio, index

    def optimize(self, max_iter=10, alpha_const=1., gradient_method=None, optimization_step='vanilla', backtracking=False, fine_tuning=False):
        f = self.eval
        grad = self.grad
        f_ratio = lambda thetas: self.eval_ratio(self.get_counts(thetas))
        num_variables = len(self.thetas_init)
        optimizer = Optimizer(f, grad, f_ratio, num_variables, alpha_const=alpha_const,
                              max_iter=max_iter, gradient_method=gradient_method, optimization_step=optimization_step,
                              backtracking=backtracking, fine_tuning=fine_tuning)
        optimizer.optimize(self.thetas_opt)
        self.thetas_opt = optimizer.x
        self.history['loss'] = self.history['loss'] + optimizer.history['loss']
        self.history['norm_grad'] = self.history['norm_grad'] + optimizer.history['norm_grad']
        self.history['ratio'] = self.history['ratio'] + optimizer.history['ratio']
        self.history['index'] = self.history['index'] + optimizer.history['index']
        self.time = self.time + optimizer.time
        return

    def get_counts(self, thetas):
        circuit = QuantumCircuit(self.n)
        circuit.append(self.rotation_circuit(thetas).to_gate(), list(range(self.n)))
        if self.entanglement is not None:
            circuit.append(self.entanglement_circuit().to_gate(), list(range(self.n)))
        circuit.barrier()
        circuit.measure_all()
        counts = self.run(circuit)
        counts = {k[::-1]: v for k, v in counts.items()}
        return counts

    def result(self):
        counts = self.get_counts(self.thetas_opt)

        index = max(counts, key=counts.get)
        print('\nmy results'.upper())
        print(self.expected_cut)
        print(f'       Obtained Cut: {self.cut(index)[0]} and {self.cut(index)[1]}')
        print(f'Approximation Ratio: {self.history["ratio"][-1]}')
        print(f'Approximation Index: {self.history["index"][-1]}')
        print(f'     Execution Time: {self.time} seconds')

        # Plot Histogramm
        plot_histogram(counts,
                       title=f'\nExpected classical optimum = {np.min(self.costs)} '
                             f'at state = {np.binary_repr(np.argmin(self.costs), width=self.n)}')
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.show()

        # draw graph
        res_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
        qcut = list(res_counts.keys())[-1]
        colors = ['y' if int(i) == 1 else 'lightblue' for i in qcut]

        self.draw_graph(colors=colors, title='\n'.join(self.expected_cut.split('\n')[:3]))
        # Plot statistics
        y = self.history['ratio']
        plt.plot(np.arange(-1, len(y)-1), y)
        plt.xlabel('Iteration')
        plt.ylabel('Approximation Ratio')
        plt.ylim(-.1, 1.1)
        plt.show()
        return counts

    def classical_cost(self, state):
        cost = 0
        for u, v in self.G.edges:
            if u == v:
                cost += self.G[u][v]['weight'] * (-1) ** int(state[u])
            else:
                cost += self.G[u][v]['weight'] * (-1) ** int(state[u]) * (-1) ** (int(state[v]))
        return cost

    def classical_expectation(self, counts):
        cost = 0
        for state, prob in counts.items():
            cost += (prob/self.shots) * self.classical_cost(state)
        return cost

    def quantum_cost(self, counts):
        cost = self.K * (counts.get('0', 0) - counts.get('1', 0)) / self.shots
        return cost

    def run(self, circuit):
        backend = Aer.get_backend('qasm_simulator')
        circuit_neu = transpile(circuit, backend=backend)
        job = backend.run(circuit_neu, shots=self.shots).result()
        counts = job.get_counts(circuit_neu)
        return counts


if __name__ == '__main__':
    plt.close('all')
    n = 5
    p = (1., 0.)
    entanglement = {1: None,
                    2: 'bell',
                    3: 'full',
                    4: 'linear',
                    5: 'circular'}
    gradient_method = {1: None,
                       2: 'parameter_shift',
                       3: 'finite_differences',
                       4: 'spsa'}
    optimization_step = {1: 'vanilla',
                         2: 'adam'}
    mc = MaxCut(n=n, p=p, entanglement=entanglement[1], brute_force=False, exact_costs=False)
    mc.draw_graph()
    mc.draw_circuit()
    mc.optimize(max_iter=20, alpha_const=1., backtracking=False, gradient_method=gradient_method[2], optimization_step=optimization_step[1], fine_tuning=False)
    mc.result()





