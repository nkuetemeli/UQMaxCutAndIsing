import matplotlib.pyplot as plt

from max_cut import *


class QAOABenchmark(MaxCut):
    def __init__(self, n, p, seed=10, params_init=None, reps=None):
        super().__init__(n=n, p=p, seed=seed, brute_force=False)

        self.reps = int(np.ceil(self.n/2)) if reps is None else reps
        self.params_init = np.zeros(2 * self.reps, ) if params_init is None else params_init
        self.params_opt = self.params_init

        self.qaoa_history = {'loss': [], 'norm_grad': [], 'ratio': [], 'index': []}
        self.qaoa_time = 0

    def entangled_rz(self, circuit, qubit_1, qubit_2, angle):
        circuit.cx(qubit_1, qubit_2)
        circuit.rz(angle, qubit_2)
        circuit.cx(qubit_1, qubit_2)
        return circuit

    def UC(self, gamma):
        circuit = QuantumCircuit(self.n, name=rf' U(C, $\gamma$) ')
        for u, v in self.G.edges:
            self.entangled_rz(circuit, qubit_1=u, qubit_2=v, angle=self.G[u][v]['weight'] * gamma)
            circuit.barrier()
        return circuit

    def UB(self, beta):
        circuit = QuantumCircuit(self.n, name=rf' U(B, $\beta$) ')
        for x in range(self.n):
            circuit.rx(beta, x)
        return circuit

    def qaoa(self, params):
        circuit = QuantumCircuit(self.n)
        register = list(range(self.n))
        circuit.h(register)

        gammas, betas = params[:self.reps], params[self.reps:]

        for gamma, beta in zip(gammas, betas):
            circuit.append(RemoveBarriers()(self.UC(gamma)).to_gate(), register)
            circuit.barrier()
            circuit.append(RemoveBarriers()(self.UB(beta)).to_gate(), register)
        circuit.measure_all()
        return circuit

    def qaoa_eval(self, params):
        counts = self.qaoa_get_counts(params)
        val = self.classical_expectation(counts)
        return val

    def qaoa_grad(self, params, iter, gradient_method=None):
        grad = Grad(f=self.qaoa_eval, x=params, iter=iter, gradient_method=gradient_method)
        return grad()

    def qaoa_optimize(self, max_iter=10, alpha_const=1., gradient_method=None, optimization_step='vanilla', backtracking=False, fine_tuning=False):
        f = self.qaoa_eval
        grad = self.qaoa_grad
        f_ratio = lambda params: self.eval_ratio(self.qaoa_get_counts(params))
        num_variables = len(self.params_init)
        optimizer = Optimizer(f, grad, f_ratio, num_variables, alpha_const,
                              max_iter=max_iter, gradient_method=gradient_method, optimization_step=optimization_step,
                              backtracking=backtracking, fine_tuning=fine_tuning)
        optimizer.optimize(self.params_opt)
        self.params_opt = optimizer.x
        self.qaoa_history['loss'] = self.qaoa_history['loss'] + optimizer.history['loss']
        self.qaoa_history['norm_grad'] = self.qaoa_history['norm_grad'] + optimizer.history['norm_grad']
        self.qaoa_history['ratio'] = self.qaoa_history['ratio'] + optimizer.history['ratio']
        self.qaoa_history['index'] = self.qaoa_history['index'] + optimizer.history['index']
        self.qaoa_time = self.qaoa_time + optimizer.time
        return

    def qaoa_get_counts(self, params):
        circuit = self.qaoa(params)
        counts = self.run(circuit)
        counts = {k[::-1]: v for k, v in counts.items()}
        return counts

    def qaoa_result(self):
        counts = self.qaoa_get_counts(self.params_opt)

        index = max(counts, key=counts.get)
        print('\nqaoa results'.upper())
        print(self.expected_cut)
        print(f'Obtained: {self.cut(index)[0]} and {self.cut(index)[1]}')
        print(f'   Ratio: {self.qaoa_history["ratio"][-1]}')
        print(f'   index: {self.qaoa_history["index"][-1]}')
        print(f'    Time: {self.qaoa_time}')

        # Plot Histogramm
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 4)
        plot_histogram(counts, ax=ax,
                       title=f'\nExpected classical optimum = {np.min(self.costs)} '
                             f'at state = {np.binary_repr(np.argmin(self.costs), width=self.n)}')
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.show()
        # Plot statistics
        y = self.qaoa_history['ratio']
        plt.plot(np.arange(-1, len(y)-1), y)
        plt.xlabel('Iteration')
        plt.ylabel('Approximation Ratio')
        plt.ylim(-.1, 1.1)
        plt.show()
        return counts

    def qaoa_draw_circuit(self):
        circuit = self.qaoa(self.params_opt)
        if self.decompose:
            circuit.decompose([rf' U(C, $\gamma$) ', rf' U(B, $\beta$) ']).draw(output='mpl')
        else:
            circuit.draw(output='mpl')
        plt.show()
        return


def qaoa_landscape():
    n = 3
    p = (1., 0.)
    n_points = 20
    x = np.linspace(0, np.pi, n_points)
    gammas, betas = np.meshgrid(x, x)

    resource = np.zeros_like(gammas)
    qaoa_bm = None
    for i in range(n_points):
        for j in range(n_points):
            params = np.array([gammas[i, j], betas[i, j]])
            qaoa_bm = QAOABenchmark(n=n, p=p, params_init=params, reps=1)
            resource[i, j] = qaoa_bm.eval_ratio(qaoa_bm.qaoa_get_counts(qaoa_bm.params_init))[0]
    qaoa_bm.qaoa_draw_circuit()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(gammas, betas, resource, cmap='Wistia')
    plt.show()


if __name__ == '__main__':
    n = 5
    p = (1., 0.)
    gradient_method = {1: None,
                       2: 'parameter_shift',
                       3: 'finite_differences',
                       4: 'spsa'}
    optimization_step = {1: 'vanilla',
                         2: 'adam'}
    # qaoa_landscape()
    qaoa_bm = QAOABenchmark(n=n, p=p)
    qaoa_bm.qaoa_draw_circuit()
    qaoa_bm.qaoa_optimize(max_iter=10, alpha_const=1., gradient_method=gradient_method[2], optimization_step=optimization_step[1])
    qaoa_bm.qaoa_result()

