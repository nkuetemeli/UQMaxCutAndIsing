import dimod
import dwave.inspector
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import time
from src.max_cut import *


class DWAVEBenchmark(MaxCut):
    def __init__(self, n, p, seed=10):
        super().__init__(n=n, p=p, seed=seed, brute_force=False)
        self.num_reads = 50
        self.dwave_history = {'loss': [], 'norm_grad': [], 'ratio': [], 'index': []}
        self.dwave_time = None
        self.classical_expectation = self.dwave_classical_expectation

    def dwave_classical_expectation(self, counts):
        cost = 0
        for state, prob in counts.items():
            cost += (prob / self.num_reads) * self.classical_cost(state)
        return cost

    def solve(self, inspector=False):
        tic = time.time()
        adjacency = self.adjacency
        # Construct BQM in QUBO-form instead of ISING
        bqm = dimod.BinaryQuadraticModel(
            {
                (i, j): 4 * adjacency[i, j] if i != j else -2 * (self.adjacency[i, i] +
                                                                 np.sum(self.adjacency[:i, i]) +
                                                                 np.sum(self.adjacency[i, i + 1:]))
                for i in range(self.n)
                for j in range(i, self.n)
            },
            offset=np.sum(np.diag(adjacency)) + np.sum(np.triu(adjacency, 1)),
            vartype=dimod.BINARY
        )

        qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})
        sampler = EmbeddingComposite(qpu_advantage)

        response = sampler.sample(bqm, num_reads=self.num_reads)
        x = ''.join([str(i) for i in list(response.first.sample.values())])

        counts = {np.binary_repr(i, width=self.n): 0 for i in range(2 ** self.n)}
        for result in response.data(['sample', 'energy', 'num_occurrences', 'chain_break_fraction']):
            sol = ''.join([str(i) for i in list(result.sample.values())])
            counts[sol] = result.num_occurrences

        ratio, index = self.eval_ratio(counts)
        self.dwave_history['loss'].append(self.classical_expectation(counts))
        self.dwave_history['norm_grad'].append(None)
        self.dwave_history['ratio'].append(ratio)
        self.dwave_history['index'].append(index)
        self.dwave_time = time.time() - tic

        index = [int(i) for i in x]
        print('\ndwave results'.upper())
        print(self.expected_cut)
        print(f'Obtained: {self.cut(index)[0]} and {self.cut(index)[1]}')
        print(f'   Ratio: {self.dwave_history["ratio"][-1]}')
        print(f'   Index: {self.dwave_history["index"][-1]}')
        print(f'    Time: {self.dwave_time}')
        if inspector:
            dwave.inspector.show(response)

        return


if __name__ == '__main__':
    n = 10
    p = (1., 0.)
    dwave_bm = DWAVEBenchmark(n=n, p=p)
    dwave_bm.solve(inspector=True)
