# Universal Quantum Algorithms <br/> For The Weighted Maximum Cut Problem And The Ising Model
This code can be used to reproduce the experiments and results of the paper
**Universal Quantum Algorithms For The Weighted Maximum Cut Problem And The Ising Model**.
$x$ with $f(x) = 12$

Given an input graph 
$ \mathcal G = (\mathcal S, \mathcal E, \mathcal{C}) $ 
with 
$ \mathcal S = \left\{s_1, \ldots, s_n\right\} $, 
$ \mathcal E \subseteq \mathcal S \times \mathcal S $ 
and 
$ \mathcal{C} : \mathcal E \to \R $ with $\mathcal{C}(s_i, s_j) =: \mathcal{C}_{ij}$ 
being a cost function on $ \mathcal E $,
the `MaxCut` class instance compute an outputs an approximate solution for the graph partitioning problem.
The output is an approximate solution of the ground state of the Ising model 
or if $\mathcal{C}_{ii} = 0$ for all $i$, that of the weighted maximum cut problem.

# Install
The code depends on the Python packages 
[numpy](https://numpy.org/install/), 
[networkx](https://networkx.org/documentation/stable/install.html), 
[qiskit](https://qiskit.org/documentation/stable/0.24/install.html) 
and [dwave](https://docs.ocean.dwavesys.com/projects/system/en/latest/installation.html).
Please refer to the product pages for reference.

Once you satisfied the dependency, download the repository and run pip install . inside the directory.

# Example

    # Run the code on a randomly generated graph 
    n = 5
    p = (1., 0.)
    entanglement = None
    gradient_method = 'parameter_shift'
    optimization_step = 'vanilla'

    mc = MaxCut(n=n, p=p, entanglement=entanglement, brute_force=True, exact_costs=True)
    mc.draw_graph()
    mc.draw_circuit()
    mc.optimize(max_iter=10, alpha_const=1., gradient_method=gradient_method, optimization_step=optimization_step)
    mc.result()

# Run demos

    # Brute force the problem and verify the correctness of the transformed costs
    # Lasts about 2 days to brute force 10 instances of a 10 node fully connected graph
    eval_mc()

    # Benchmarking UQMC with QAOA and D-Wave
    # Lasts about 3 hours to benchmark 20 instances of 3, 5 and 10 node fully connected graphs
    benchmark_mc(file_name='experiment_benchmark_mc')

    # Benchmarking UQIM with D-Wave
    # Lasts about 2 hours to benchmark 20 instances of 3, 5 and 10 node fully connected graphs
    benchmark_ising(file_name='experiment_benchmark_ising')

# Visualize demos
    # Visualize the brute force experiment
    vis_eval_mc(save=False)

    # Visualize the MaxCut benchmark experiment
    vis_benchmark_mc(file_name='experiment_benchmark_mc', save=False)
    
    # Visualize the Ising benchmark experiment
    vis_benchmark_ising(file_name='experiment_benchmark_ising', save=False)

# Citation
If you find this work useful, please cite the article [Article URL](#).
