from qiskit_aqua.utils import split_dataset_to_data_and_labels
from qiskit_aqua.input import SVMInput
from qiskit_qcgpu_provider import QCGPUProvider
from qiskit_aqua import run_algorithm
n = 2 # How many features to use (dimensionality)
training_dataset_size = 20
testing_dataset_size = 10

sample_Total, training_input, test_input, class_labels = breast_cancer(training_dataset_size, testing_dataset_size, n)

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
print(class_to_label)
params = {
   'problem': {'name': 'svm_classification', 'random_seed': 10598},
   'algorithm': { 'name': 'QSVM.Kernel' },
   'backend': {'name': 'qasm_simulator', 'shots': 1024},
   'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement': 'linear'}
}

backend = QCGPUProvider().get_backend('qasm_simulator')

algo_input = SVMInput(training_input, test_input, datapoints[0])
%time result = run_algorithm(params, algo_input)
%time result = run_algorithm(params, algo_input, backend=backend)

print("ground truth:    {}".format(datapoints[1]))
print("prediction:      {}".format(result['predicted_labels']))
