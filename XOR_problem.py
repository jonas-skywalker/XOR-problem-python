import neuralnetwork
import random
import my_matrix_lib as matrix

outputs = []


data_set = [[[0.01, 0.01], [0.01]],
            [[0.99, 0.99], [0.01]],
            [[0.99, 0.01], [0.99]],
            [[0.01, 0.99], [0.99]]]

data_set2 = [[[0.0, 0.0], [0.0]],
             [[1.0, 1.0], [0.0]],
             [[1.0, 0.0], [1.0]],
             [[0.0, 1.0], [1.0]]]


nn = neuralnetwork.NeuralNetwork([2, 2, 1], 0.2)


for i in range(50000):
    data = random.choice(data_set2)
    input_list = data[0]
    target_list = data[1]
    nn.linear_regression_gradient_descent(input_list, target_list)


outputs.append(nn.feed_forward([0.0, 0.0]))
outputs.append(nn.feed_forward([1.0, 1.0]))
outputs.append(nn.feed_forward([1.0, 0.0]))
outputs.append(nn.feed_forward([0.0, 1.0]))

for mat in outputs:
    matrix.Matrix.print_mat(mat)

nn.save_json("XOR_data.json")

