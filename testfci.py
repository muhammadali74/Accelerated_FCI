# import sys
# sys.path.append('..')
import timeit

import random
import numpy as np
from causal_discovery_algs import LearnStructFCI
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr  # import a CI test that estimates partial correlation
from experiment_utils.synthetic_graphs import create_random_dag_with_latents, sample_data_from_dag
from causal_discovery_utils.performance_measures import calc_structural_accuracy_pag, find_true_pag
from matplotlib import pyplot as plt

rand_seed = 42  # arbitrary seed (ASCII code of the Asterisk symbol)
random.seed(rand_seed)
np.random.seed(rand_seed)

num_nodes = 15
num_records = 1000
connectivity_coeff = 2.0
min_lin_coeff = 0.5  # minimal 'strength' of an edge
max_lin_coeff = 2.0  # maximal 'strength' of an edge
alpha = 0.01


# curating data and output and true DAG
dag, observed_set, latents_set = create_random_dag_with_latents(
    num_nodes,
    connectivity_coeff
)

true_pag = find_true_pag(dag, observed_set)

dataset = sample_data_from_dag(
    dag,
    num_records,
    min_edge_weight=min_lin_coeff, max_edge_weight=max_lin_coeff
)

# CI test
par_corr_icd = CondIndepParCorr(
    dataset=dataset,
    threshold=alpha,
    count_tests=True,
    use_cache=True
)

print(dataset.shape)

# run fci on dataset
# Learn the PAG
par_corr_fci = CondIndepParCorr(dataset=dataset, threshold=alpha, count_tests=True, use_cache=True)  # CI test
print(par_corr_fci)
fci = LearnStructFCI(observed_set, par_corr_fci)  # instantiate an FCI learner
print(timeit.timeit(lambda:fci.learn_structure(),number = 1)) # learn the PAG

learned_pag_fci = fci.graph
print(fci.graph)

# Calculate structural errors: FCI algorithm
fci_result = calc_structural_accuracy_pag(pag_tested=learned_pag_fci, pag_correct=true_pag)

def print_structural_accuracy(structural_accuracy: dict):
    print('Edge precision: {:.2f}'.format(structural_accuracy['edge_precision']))
    print('Edge recall: {:.2f}'.format(structural_accuracy['edge_recall']))
    print('F1 Score: {:.2f}'.format(structural_accuracy['edge_F1']))
    print('Orientation accuracy: {:.2f}'.format(structural_accuracy['orientation_correctness']))



print('\nFCI performance')
print('---------------')
print_structural_accuracy(fci_result)
print('Total number of CI tests: ', sum(par_corr_fci.test_counter))

plt.figure()
x_range = range(len(par_corr_icd.test_counter))
plt.plot(x_range, par_corr_fci.test_counter, 'xr:')
plt.legend(('FCI'))
plt.xlabel('conditioning set size')
plt.ylabel('number of CI tests')
plt.grid(True)
plt.show()

