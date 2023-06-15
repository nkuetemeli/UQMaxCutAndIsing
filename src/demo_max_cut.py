from src.max_cut import *
from src.qaoa_bm import *
from src.dwave_bm import *
from os import path
import json

results_folder = '../results/'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def eval_mc():
    p = (1., 1.)
    brute_force = True

    thetas_init = None
    entanglement = None
    decompose = None
    exact_costs = False

    def experiment_coef_K(file_name, weight_bounds, shots=1024):
        """
        Experiments on the impact for the variable K
        Results are saved in the form of a dictionary {n:
                                                        {coef_K:
                                                            {costs: [[simulation 1], ..., [simulation n]],
                                                            acosts: [[simulation 1], ..., [simulation n]],
                                                            qcosts: [[simulation 1], ..., [simulation n]]}}
        """
        file_name = path.join(results_folder, file_name)
        ns = [10, ]
        coef_Ks = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
        results = {n: {k: None for k in coef_Ks} for n in ns}
        num_simulations = 10

        for n in ns:
            for coef_K in coef_Ks:
                res = {'costs': [], 'acosts': [], 'qcosts': []}
                for sim in range(num_simulations):
                    seed = sim
                    mc = MaxCut(
                        n=n, p=p, seed=seed, weight_bounds=weight_bounds,
                        shots=shots,
                        coef_K=coef_K, thetas_init=thetas_init,
                        brute_force=brute_force, exact_costs=exact_costs,
                        entanglement=entanglement, decompose=decompose
                    )
                    res['costs'].append(mc.costs)
                    res['acosts'].append(mc.acosts)
                    res['qcosts'].append(mc.qcosts)
                results[n][coef_K] = res

        with open(file_name, 'w') as file:
            file.write(json.dumps(results, cls=NpEncoder))  # use `json.loads` to do the reverse

        return

    def experiment_num_shots(file_name, weight_bounds, coef_K=.5):
        """
        Experiments on the impact for the number of shots relative to number of variables
        Results are saved in the form of a dictionary {n:
                                                        {shots:
                                                            {costs: [[simulation 1], ..., [simulation n]],
                                                            acosts: [[simulation 1], ..., [simulation n]],
                                                            qcosts: [[simulation 1], ..., [simulation n]]}}
        """
        file_name = path.join(results_folder, file_name)
        ns = [10, ]
        num_shots = [128, 256, 512, 1024]
        results = {n: {shots: None for shots in num_shots} for n in ns}
        num_simulations = 10

        for n in ns:
            for shots in num_shots:
                res = {'costs': [], 'acosts': [], 'qcosts': []}
                for sim in range(num_simulations):
                    seed = sim
                    mc = MaxCut(
                        n=n, p=p, seed=seed, weight_bounds=weight_bounds,
                        shots=shots,
                        coef_K=coef_K, thetas_init=thetas_init,
                        brute_force=brute_force, exact_costs=exact_costs,
                        entanglement=entanglement, decompose=decompose
                    )
                    res['costs'].append(mc.costs)
                    res['acosts'].append(mc.acosts)
                    res['qcosts'].append(mc.qcosts)
                results[n][shots] = res

        with open(file_name, 'w') as file:
            file.write(json.dumps(results, cls=NpEncoder))  # use `json.loads` to do the reverse
        return

    for experiment_name, weight_bounds in zip(['positive_weights', 'positive_negative_weights'], [(1, 10), (-10, 10)]):
        experiment_coef_K(file_name='experiment_coef_K_' + experiment_name+'.json', weight_bounds=weight_bounds)
        experiment_num_shots(file_name='experiment_num_shots_' + experiment_name+'.json', weight_bounds=weight_bounds)
    return


def benchmark_mc(file_name):
    """
    Benchmarks UQIM with QAOA and D-Wave
    Results are saved in the form of a dictionary {n:
                                                    ratio:
                                                        {qaoa_gradient: [[simulation 1], ..., [simulation n]],
                                                         qaoa_black_box: [[simulation 1], ..., [simulation n]],
                                                         dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    index:
                                                        {qaoa_gradient: [[simulation 1], ..., [simulation n]],
                                                         qaoa_black_box: [[simulation 1], ..., [simulation n]],
                                                         dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    time:
                                                        {qaoa_gradient: [[simulation 1], ..., [simulation n]],
                                                         qaoa_black_box: [[simulation 1], ..., [simulation n]],
                                                         dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]}}
    """
    file_name = path.join(results_folder, file_name)
    ns = [3, 5, 10, ]
    p = (1., 0.)
    results = {n: {'ratio': None, 'index': None, 'time': None} for n in ns}
    num_simulations = 20
    entanglements = {1: None,
                    2: 'bell',
                    3: 'full',
                    4: 'linear',
                    5: 'circular'}
    gradient_methods = {1: None,
                        2: 'parameter_shift',
                        3: 'finite_differences',
                        4: 'spsa'}
    optimization_steps = {1: 'vanilla',
                          2: 'adam'}

    max_iter = 20
    entanglement = entanglements[1]
    gradient_method = gradient_methods[2]
    optimization_step = optimization_steps[1]
    backtraking = False

    alpha_const_qaoa = 2.
    alpha_const_mc = 2.
    for n in ns:
        ratio = {'qaoa_gradient': [], 'qaoa_black_box': [], 'dwave': [], 'mc': []}
        index = {'qaoa_gradient': [], 'qaoa_black_box': [], 'dwave': [], 'mc': []}
        time = {'qaoa_gradient': [], 'qaoa_black_box': [], 'dwave': [], 'mc': []}
        for sim in range(num_simulations):
            seed = sim

            qaoa_bm_gradient = QAOABenchmark(n=n, p=p, seed=seed)
            qaoa_bm_gradient.qaoa_optimize(max_iter=max_iter, backtracking=backtraking, alpha_const=alpha_const_qaoa, gradient_method=gradient_method, optimization_step=optimization_step)

            qaoa_bm_black_box = QAOABenchmark(n=n, p=p, seed=seed)
            qaoa_bm_black_box.qaoa_optimize(max_iter=max_iter, alpha_const=alpha_const_qaoa, backtracking=backtraking, gradient_method=None, optimization_step=optimization_step)

            dwave_bm = DWAVEBenchmark(n=n, p=p, seed=seed)
            dwave_bm.solve(inspector=False)

            mc = MaxCut(n=n, p=p, entanglement=entanglement, seed=seed)
            mc.optimize(max_iter=max_iter, alpha_const=alpha_const_mc, backtracking=backtraking, gradient_method=gradient_method, optimization_step=optimization_step)

            ratio['qaoa_gradient'].append(qaoa_bm_gradient.qaoa_history['ratio'])
            ratio['qaoa_black_box'].append(qaoa_bm_black_box.qaoa_history['ratio'])
            ratio['dwave'].append(dwave_bm.dwave_history['ratio'])
            ratio['mc'].append(mc.history['ratio'])

            index['qaoa_gradient'].append(np.array(qaoa_bm_gradient.qaoa_history['index']).astype(np.int32))
            index['qaoa_black_box'].append(np.array(qaoa_bm_black_box.qaoa_history['index']).astype(np.int32))
            index['dwave'].append(np.array(dwave_bm.dwave_history['index']).astype(np.int32))
            index['mc'].append(np.array(mc.history['index']).astype(np.int32))

            time['qaoa_gradient'].append(qaoa_bm_gradient.qaoa_time)
            time['qaoa_black_box'].append(qaoa_bm_black_box.qaoa_time)
            time['dwave'].append(dwave_bm.dwave_time)
            time['mc'].append(mc.time)

        results[n]['ratio'] = ratio
        results[n]['index'] = index
        results[n]['time'] = time

    with open(file_name, 'w') as file:
        file.write(json.dumps(results, cls=NpEncoder))  # use `json.loads` to do the reverse
    return



def benchmark_ising(file_name):
    """
    Benchmarks UQIM with QAOA and D-Wave
    Results are saved in the form of a dictionary {n:
                                                    ratio:
                                                        {dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    index:
                                                        {dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]},
                                                    time:
                                                        {dwave: [[simulation 1], ..., [simulation n]],
                                                         mc: [[simulation 1], ..., [simulation n]]}}
    """
    file_name = path.join(results_folder, file_name)
    ns = [3, 5, 10, ]
    p = (1., 1.)
    results = {n: {'ratio': None, 'index': None, 'time': None} for n in ns}
    num_simulations = 20
    entanglements = {1: None,
                    2: 'bell',
                    3: 'full',
                    4: 'linear',
                    5: 'circular'}
    gradient_methods = {1: None,
                        2: 'parameter_shift',
                        3: 'finite_differences',
                        4: 'spsa'}
    optimization_steps = {1: 'vanilla',
                          2: 'adam'}

    max_iter = 20
    entanglement = entanglements[1]
    gradient_method = gradient_methods[2]
    optimization_step = optimization_steps[1]
    backtracking = False

    alpha_const_mc = 2.
    for n in ns:
        ratio = {'dwave': [], 'mc': []}
        index = {'dwave': [], 'mc': []}
        time = {'dwave': [], 'mc': []}
        thetas_init = np.pi/2 + np.zeros(n, )
        for sim in range(num_simulations):
            seed = sim

            dwave_bm = DWAVEBenchmark(n=n, p=p, seed=seed)
            dwave_bm.solve(inspector=False)

            mc = MaxCut(n=n, p=p, thetas_init=thetas_init, entanglement=entanglement, seed=seed)
            mc.optimize(max_iter=max_iter, alpha_const=alpha_const_mc, backtracking=backtracking, gradient_method=gradient_method, optimization_step=optimization_step)

            ratio['dwave'].append(dwave_bm.dwave_history['ratio'])
            ratio['mc'].append(mc.history['ratio'])

            index['dwave'].append(np.array(dwave_bm.dwave_history['index']).astype(np.int32))
            index['mc'].append(np.array(mc.history['index']).astype(np.int32))

            time['dwave'].append(dwave_bm.dwave_time)
            time['mc'].append(mc.time)

        results[n]['ratio'] = ratio
        results[n]['index'] = index
        results[n]['time'] = time

    with open(file_name, 'w') as file:
        file.write(json.dumps(results, cls=NpEncoder))  # use `json.loads` to do the reverse
    return


def main():
    pass


if __name__ == '__main__':
    # Brute force the problem and verify the correctness of the transformed costs
    # Lasts about 2 days to brute force 10 instances of a 10 node fully connected graph
    # eval_mc()

    # Benchmarking UQMC with QAOA and D-Wave
    # Lasts about 3 hours to benchmark 20 instances of 3, 5 and 10 node fully connected graphs
    benchmark_mc(file_name='experiment_benchmark_mc')

    # Benchmarking UQIM with D-Wave
    # Lasts about 2 hours to benchmark 20 instances of 3, 5 and 10 node fully connected graphs
    benchmark_ising(file_name='experiment_benchmark_ising')
    pass
