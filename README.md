# A Universal Quantum Algorithm For MaxCut And Ising
This code can be used to reproduce the experiments and results of the paper <br/>
**A Universal Quantum Algorithm for Weighted Maximum Cut and Ising Problems**.

    Kuete Meli, N., Mannel, F. & Lellmann, J. 
    A universal quantum algorithm for weighted maximum cut and Ising problems. 
    Quantum Inf Process 22, 279 (2023). https://doi.org/10.1007/s11128-023-04025-x

Given an input graph, the `MaxCut` class instance computes and outputs an approximate solution for the graph partitioning problem.
A demo Notebook is provided in the file `DEMO.ipynb`.

# Install
The code depends on the Python packages 
[numpy](https://numpy.org/install/), 
[networkx](https://networkx.org/documentation/stable/install.html), 
[qiskit](https://qiskit.org/documentation/stable/0.24/install.html) 
and [dwave](https://docs.ocean.dwavesys.com/projects/system/en/latest/installation.html).

- Please download the repository and install the requirements in `requirements.txt` or refer to the product pages for reference.

- Once you satisfied the dependency, run `pip install .` inside the directory.

Move to the `src` folder to run the subsequent commands.

# Example

    # Run the code on a randomly generated graph
    # Close the Plots as they pop-up to continue the execution of the script
    python max_cut.py

# Run demos

    # Brute force the problem and verify the correctness of the transformed costs
    # May last several hours: Consider in-comment in the file the experiment your are interested in.
    python demo_max_cut.py

# Visualize demos
    # Visualize the brute force experiment
    python vis_max_cut.py

# Citation
If you find this work useful, please cite the article [Article URL](https://doi.org/10.1007/s11128-023-04025-x).
