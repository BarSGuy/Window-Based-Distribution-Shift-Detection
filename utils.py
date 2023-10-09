import os
import pickle
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import math
from easydict import EasyDict as edict

matplotlib.use('Agg')
import torch
import random
from metrics import *
import torch.nn.functional as F

import os
import csv

import torch


# ================================= general ================================= #


def print_nested_dict(nested_dict):
    pprint.pprint(nested_dict)


def count_duplicate_tensors(tensor_list):
    # Convert tensors to numpy arrays
    array_list = [tensor.numpy() for tensor in tensor_list]

    # Find unique elements in the list
    unique_array_list, unique_counts = np.unique(array_list, axis=0, return_counts=True)

    # Check if there are duplicates
    num_duplicates = np.sum(unique_counts > 1)

    if num_duplicates > 0:
        print(f"The list contains {num_duplicates} tensors that are duplicates.")
    else:
        print("The list does not contain any duplicate tensors.")


def includes_zero_and_one(arr):
    """
    Checks if a binary array includes both 0 and 1 in its values.

    Args:
    - arr: a binary array, represented as a list of integers (0 or 1).

    Returns:
    - True if the array includes both 0 and 1, False otherwise.
    """
    if 0 in arr and 1 in arr:
        return True
    else:
        return False


def softmax(logits, temprature=1.0):
    # Check if input tensor is on a GPU device
    device = logits.device
    # Apply the softmax function to the logits tensor
    # softmax_values = torch.softmax(logits, dim=0)
    # torch.tensor(logits_list)
    softmax_values = F.softmax(logits / temprature, dim=0)
    # Return the softmax tensor on the same device as input tensor
    return softmax_values.to(device)


def get_bootstrapped_metrics_with_stds(num_bootstrap_runs, all_preds, all_labels, threshold=0.95):
    N = len(all_labels)
    bootstrap_stats = []
    for i in range(num_bootstrap_runs):

        indices = [random.randint(0, N - 1) for i in range(N)]
        all_preds_bootstrap_sample = all_preds[indices]
        all_labels_bootstrap_sample = all_labels[indices]
        if not includes_zero_and_one(all_labels_bootstrap_sample):
            continue
        bootstrap_statistic = calc_metrics(predictions=all_preds_bootstrap_sample,
                                           labels=all_labels_bootstrap_sample, threshold=threshold)
        bootstrap_stats.append(bootstrap_statistic)
    final_dict = compute_bootstrap_stats(bootstrap_stats)
    return final_dict


def save_dict_to_csv(data_dict, filename):
    """
    Save a dictionary to a CSV file.

    Args:
    - data_dict: a dictionary containing the mean and standard deviation of each metric across all the bootstrap samples.
    - filename: the name (and path) of the CSV file to save the data to.

    Returns:
    - None
    """
    # Extract the directory path and create it if it doesn't exist
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        header = ['Metric', 'Mean', 'Standard Deviation']
        writer.writerow(header)

        # Write the data rows
        for key, value in data_dict.items():
            row = [key, value['mean'], value['std']]
            writer.writerow(row)

import csv
import os

def append_csv(row_name, value, file_name):
    # Check if file exists
    file_exists = os.path.isfile(file_name)

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    # Create file if it doesn't exist
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Check if row already exists
        if not file_exists or row_name not in open(file_name).read():
            writer.writerow([row_name, value])
            print(f"Added {row_name} with value {value} to {file_name}")
        else:
            print(f"{row_name} already exists in {file_name}")


def compute_bootstrap_stats(bootstrap_stats):
    """
    Computes the mean and standard deviation of each metric across all the bootstrap samples.

    Args:
    - bootstrap_stats: a list of dictionaries, each representing several metrics.

    Returns:
    - A dictionary containing the mean and standard deviation of each metric across all the bootstrap samples.
    """
    # Extract the keys of the first dictionary in the list
    keys = list(bootstrap_stats[0].keys())

    # Initialize a dictionary to store the aggregated statistics
    agg_stats = {key: [] for key in keys}

    # Loop over the bootstrap samples and compute the statistics for each metric
    for sample in bootstrap_stats:
        for key in keys:
            agg_stats[key].append(sample[key])

    # Compute the mean and standard deviation for each metric
    mean_stats = {key: np.mean(agg_stats[key]) for key in keys}
    std_stats = {key: np.std(agg_stats[key], ddof=1) for key in keys}

    # Combine the mean and standard deviation for each metric into a single dictionary
    result_dict = {key: {
        'mean': mean_stats[key],
        'std': std_stats[key]} for key in keys}

    return result_dict


def max_lists(list1, list2):
    """
    Given two lists of the same size, returns a new list containing the maximum value at each position.
    """
    return [max(list1[i], list2[i]) for i in range(len(list1))]


def concatenate_lists(list1, list2):
    """
    Concatenates two lists and returns the result as a new list.
    """
    return list1 + list2


def split_array(arr, x):
    N = len(arr)
    indices = list(range(N))
    random.shuffle(indices)
    return arr[indices[:int(x)]], arr[indices[int(x):]]


def shuffle_list(lst):
    """Returns a shuffled list."""
    shuffled_lst = lst[:]  # make a copy of the original list to avoid modifying it
    random.shuffle(shuffled_lst)
    return shuffled_lst


def shuffle_tensor_rows(tensor):
    """Returns a tensor with shuffled rows along the first dimension."""
    indices = list(range(tensor.size(0)))
    random.shuffle(indices)
    shuffled_tensor = tensor[indices]
    return shuffled_tensor


def save_data_to_pickle_file(file_path: str, data: object) -> None:
    """
    Save data to a pickle file at the specified file path.

    If the file path does not exist, it will be created.

    Parameters
    ----------
    file_path : str
        The path to save the file to, including the file name.
    data : object
        The data to save.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If file_path is not a string, or if data cannot be pickled.
    """
    try:
        # Create directory if it does not exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write data to file using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    except (TypeError, pickle.PicklingError) as e:
        raise TypeError("Failed to save data to pickle file") from e


def load_pickle_file(file_path):
    """
    Loads a pickle file from the specified file path.

    Args:
        file_path (str): Path to the pickle file to be loaded, including the filename.

    Returns:
        The contents of the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def plot_bounds(cfg, should_hold, upper_bounds, window_size, lower_bounds, actual_coverage_for_lower,
                actual_coverage_for_upper,
                desired_coverages, suffix_for_file_name=''):
    """
    This function plots the upper and lower bounds as well as the actual coverage for each coverage parameter. The desired
    coverages are also plotted as scatter points. Violations of the upper and lower bounds are marked as red and blue
    circles, respectively.

    Args:
        cfg: A configuration object
        should_hold (bool): Should the upper and lower bounds
        window_size (int): The window sizes
        upper_bounds (list or array): The upper bounds for each coverage parameter - function of the in-distribution dataset only!
        lower_bounds (list or array): The lower bounds for each coverage parameter - function of the in-distribution dataset only!
        actual_coverage_for_lower (list or array): The actual coverage for each coverage parameter for the lower bound - function of the out-of-distribution dataset only!
        actual_coverage_for_upper (list or array): The actual coverage for each coverage parameter for the upper bound - function of the out-of-distribution dataset only!
        desired_coverages (list or array): The desired coverage for each coverage parameter - user defined.

    Returns:
        A plot of the bounds and actual coverage

    """

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the x-axis
    x = desired_coverages

    # Compute the mean and standard deviation for each parameter
    actual_coverage_for_upper_mean = np.mean(actual_coverage_for_upper, axis=0)
    actual_coverage_for_upper_std = np.std(actual_coverage_for_upper, axis=0) / np.sqrt(len(actual_coverage_for_upper))
    actual_coverage_for_lower_mean = np.mean(actual_coverage_for_lower, axis=0)
    actual_coverage_for_lower_std = np.std(actual_coverage_for_lower, axis=0) / np.sqrt(len(actual_coverage_for_upper))

    upper_bounds = np.mean(upper_bounds, axis=0)
    lower_bounds = np.mean(lower_bounds, axis=0)
    # Plot the upper bounds and actual coverage for upper
    ax.plot(x, upper_bounds, color='blue', linestyle='dashed', label='Upper Bound', linewidth=3, alpha=0.8)
    ax.plot(x, actual_coverage_for_upper_mean, color='blue', label='Actual Coverage for Upper')
    ax.fill_between(x, actual_coverage_for_upper_mean - actual_coverage_for_upper_std,
                    actual_coverage_for_upper_mean + actual_coverage_for_upper_std,
                    color='blue', alpha=0.2)
    # Plot the lower bounds and actual coverage for lower
    ax.plot(x, lower_bounds, color='red', linestyle='solid', label='Lower Bound', linewidth=3, alpha=0.3)
    ax.plot(x, actual_coverage_for_lower_mean, color='red', label='Actual Coverage for Lower')
    ax.fill_between(x, actual_coverage_for_lower_mean - actual_coverage_for_lower_std,
                    actual_coverage_for_lower_mean + actual_coverage_for_lower_std,
                    color='red', alpha=0.2)
    # Plot the desired coverages
    ax.scatter(x, desired_coverages, color='black', marker='x', s=100, label='Desired Coverage')

    # Find the indices of the violations
    violated_lower_indices = actual_coverage_for_lower_mean < lower_bounds
    violated_upper_indices = actual_coverage_for_upper_mean > upper_bounds

    # Plot the violations of the lower bound
    if any(violated_lower_indices):
        ax.scatter(np.array(x)[violated_lower_indices],
                   np.array(actual_coverage_for_lower_mean)[violated_lower_indices],
                   color='red', marker='o', s=100, label='Violations of Lower Bound')

    # Plot the violations of the upper bound
    if any(violated_upper_indices):
        ax.scatter(np.array(x)[violated_upper_indices],
                   np.array(actual_coverage_for_upper_mean)[violated_upper_indices],
                   color='blue', marker='o', s=100, label='Violations of Upper Bound')

    # add title and legend
    plt.title('Bounds and Coverage', fontsize=18)
    plt.legend(fontsize=10)

    # set x and y axis labels and limits
    plt.xlabel('Desired Coverages', fontsize=14)
    plt.ylabel('Coverage', fontsize=14)
    EPS = 0.01
    ymin = np.min([np.min(upper_bounds), np.min(lower_bounds), np.min(actual_coverage_for_lower),
                   np.min(actual_coverage_for_upper), np.min(desired_coverages)])
    ymax = np.max([np.max(upper_bounds), np.max(lower_bounds), np.max(actual_coverage_for_lower),
                   np.max(actual_coverage_for_upper), np.max(desired_coverages)])
    plt.ylim(ymin - EPS, ymax + EPS)
    # add grid
    plt.grid(True, linestyle='--')
    # set x-axis tick locations and labels
    plt.xticks(desired_coverages)
    plt.yticks(desired_coverages)

    # Save the plot as a PNG file
    filename = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        filename = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        filename = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        filename = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}'
    if not os.path.exists(filename):
        os.makedirs(filename)
    if should_hold:
        plt.savefig(
            filename + f'/val_window_{window_size}' + suffix_for_file_name + '.png',
            dpi=300, bbox_inches='tight')
    else:
        plt.savefig(
            filename + f'/window_{window_size}' + suffix_for_file_name + '.png',
            dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def calculate_averages(parent_path, file_name, new_file_name):
    # Get a list of all child directories
    child_dirs = [os.path.join(parent_path, child) for child in os.listdir(parent_path) if
                  os.path.isdir(os.path.join(parent_path, child))]

    # Create an empty list to hold the data from all files
    data = []

    # Loop through each child directory
    for child_dir in child_dirs:
        # Construct the full path to the file in this child directory
        file_path = os.path.join(child_dir, file_name)

        # Read the data from the file and append it to the list
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

    # Calculate the averages and standard deviations
    avg_dict = edict({
        'fpr_at_95_tpr_out': edict({
            'mean': [],
            'std': []
        }),

        'fpr_at_95_tpr_in': edict({
            'mean': [],
            'std': []
        }),
        'detection_error_out': edict({
            'mean': [],
            'std': []
        }),
        'detection_error_in': edict({
            'mean': [],
            'std': []
        }),
        'auroc_out': edict({
            'mean': [],
            'std': []
        }),
        'aupr_out': edict({
            'mean': [],
            'std': []
        }),
        'aupr_in': edict({
            'mean': [],
            'std': []
        }),
        'precision_out': edict({
            'mean': [],
            'std': []
        }),
        'precision_in': edict({
            'mean': [],
            'std': []
        }),
        'recall_out': edict({
            'mean': [],
            'std': []
        }),
        'specificity_out': edict({
            'mean': [],
            'std': []
        }),
        'f1_out': edict({
            'mean': [],
            'std': []
        }),
        'f1_in': edict({
            'mean': [],
            'std': []
        }),
        'accuracy': edict({
            'mean': [],
            'std': []
        }),
    })

    for dct in data:
        avg_dict[dct['Metric']]['mean'].append(float(dct['Mean']))
        avg_dict[dct['Metric']]['std'].append(float(dct['Standard Deviation']))
    final_avg_dict = edict({
        'fpr_at_95_tpr_out': edict({
            'mean': 0,
            'std': 0,
        }),

        'fpr_at_95_tpr_in': edict({
            'mean': 0,
            'std': 0,
        }),
        'detection_error_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'detection_error_in': edict({
            'mean': 0,
            'std': 0,
        }),
        'auroc_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'aupr_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'aupr_in': edict({
            'mean': 0,
            'std': 0,
        }),
        'precision_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'precision_in': edict({
            'mean': 0,
            'std': 0,
        }),
        'recall_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'specificity_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'f1_out': edict({
            'mean': 0,
            'std': 0,
        }),
        'f1_in': edict({
            'mean': 0,
            'std': 0,
        }),
        'accuracy': edict({
            'mean': 0,
            'std': 0,
        }),
    })
    for key, val in avg_dict.items():
        metric = key
        mean_of_means = sum(val['mean']) / len(val['mean'])
        mean_of_std = math.sqrt(sum([std**2 for std in val['std']]))
        final_avg_dict[metric]['mean'] = mean_of_means
        final_avg_dict[metric]['std'] = mean_of_std
    save_dict_to_csv(data_dict=final_avg_dict, filename=parent_path + '/'+ new_file_name)

# ================================= dictionary handler ================================= #


def get_value_from_dict(data_dict, path):
    """
    Returns the value in a nested dictionary at a specified path.

    Parameters:
    - data_dict (dict): A nested dictionary to search for a value
    - path (list): A list representing the path to the value in the dictionary

    Returns:
    - The value at the specified path in the dictionary
    """
    current = data_dict
    for key in path:
        if isinstance(current, list):
            current = current[int(key)]
        else:
            current = current.get(key)
        if current is None:
            break
    return current


def get_values_from_list_dict(list_data_dict, path):
    """
    Returns the value in a nested dictionary at a specified path.

    Parameters:
    - list_data_dict (list): A list of nested dictionaries to search for a value
    - path (list): A list representing the path to the value in the dictionary

    Returns:
    - The value at the specified path in the dictionary
    """
    values = []
    for data_dict in list_data_dict:
        current = data_dict
        for key in path:
            if isinstance(current, list):
                current = current[int(key)]
            else:
                current = current.get(key)
            if current is None:
                break
        values.append(current)
    return values
