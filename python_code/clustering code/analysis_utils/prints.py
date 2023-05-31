import torch


def print_norm_for_weight(layer_name, sum_of_weights, models_weights_diff, names):
    print(f'\nLayer {layer_name}')
    print(f'sum norm: {torch.linalg.norm(sum_of_weights[layer_name])}')
    sum = 0
    for i, model_weights in enumerate(models_weights_diff):
        norma = torch.linalg.norm(model_weights[layer_name])
        sum += norma
        print(f'model {i} ({names[layer_name]}) norm: {norma}')
    print(f'Sum of all models norms {sum}\n')


def print_sim_dim_matrix(layer_number, sim_mat, names, models_weights_diff, models_names):
    print(
        f'\nCosine similarity or euclidean matrix for Layer {layer_number} ({names[layer_number]}) dim '
        f'{models_weights_diff[0][layer_number].shape}')

    for model_name in models_names:
        print(model_name, end='\t\t')
    print()

    for i in range(len(sim_mat[0][layer_number])):
        for k in range(len(sim_mat[0][i])):
            print(f'{sim_mat[layer_number][i][k].item():.2f}', end='\t\t')
        print()
