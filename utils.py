import os
import pickle
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import math
from easydict import EasyDict as edict
from methods import *
matplotlib.use('Agg')
import torch
import random
from metrics import *
import torch.nn.functional as F

import os
import csv

import torch


# ====================================================================================== #
#                                  General helpers                                       #
# ====================================================================================== #
def parse_args():
    parser = argparse.ArgumentParser(description='Script to process arguments')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to configuration file')
    parser.add_argument('--IMAGENET.GET_ACCURACY_ONLY', type=bool, default=None,
                        help='To get accuracy only')
    parser.add_argument('--IMAGENET.DATASETS.PATH_TO_IMAGENET', type=str, default=None, help='Path to ImageNet dataset')
    parser.add_argument('--IMAGENET.DATASETS.PATH_TO_IMAGENET_O', type=str, default=None,
                        help='Path to ImageNet-O dataset')
    parser.add_argument('--IMAGENET.DATASETS.PATH_TO_IMAGENET_A', type=str, default=None,
                        help='Path to ImageNet-A dataset')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS', type=float, default=None,
                        help='FGSM epsilon value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE', type=int, default=None,
                        help='FGSM batch size')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.C', type=float, default=None, help='CW C value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA', type=float, default=None, help='CW kappa value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS', type=int, default=None, help='CW steps')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.LR', type=float, default=None, help='CW learning rate')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS', type=float, default=None,
                        help='PGD epsilon value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS', type=int, default=None, help='PGD steps')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA', type=float, default=None,
                        help='PGD alpha value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START', type=bool, default=None,
                        help='PGD random start')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD', type=float, default=None,
                        help='Gaussian noise std')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE', type=int, default=None,
                        help='Rotation angle')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR', type=float, default=None, help='Zoom factor')
    parser.add_argument('--IMAGENET.EXPERIMENT.SEED', type=int, default=None, help='Experiment seed')
    parser.add_argument('--IMAGENET.EXPERIMENT.OOD', type=str, default=None, help='Out-of-distribution dataset name')
    parser.add_argument('--IMAGENET.EXPERIMENT.MMD_FIT_SIZE', type=int, default=None, help='MMD fit size')
    parser.add_argument('--IMAGENET.EXPERIMENT.KS_FIT_SIZE', type=int, default=None, help='KS fit size')
    parser.add_argument('--IMAGENET.EXPERIMENT.TRAIN_SIZE', type=int, default=None, help='Training size')
    parser.add_argument('--IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS', type=int, default=None, help='Number of random runs')
    parser.add_argument('--IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS', type=int, default=None,
                        help='Number of bootstrap runs')
    parser.add_argument('--IMAGENET.EXPERIMENT.WINDOW_SIZES', nargs='+', type=int, default=None, help='Window sizes')
    parser.add_argument('--IMAGENET.EXPERIMENT.DETECTORS', nargs='+', default=None, help='Detectors')
    parser.add_argument('--IMAGENET.EXPERIMENT.PATH_TO_RESULTS', type=str, default=None,
                        help='Path to results directory')
    parser.add_argument('--IMAGENET.MODEL', type=str, default=None, help='Model name')
    parser.add_argument('--IMAGENET.PATH_TO_SAVE_OUTPUTS', type=str, default=None, help='Path to save outputs')
    parser.add_argument('--IMAGENET.BATCH_SIZE', type=int, default=None, help='Batch size')
    parser.add_argument('--IMAGENET.PATH_TO_SAVE_ACCURACIES', type=str, default=None, help='Path to save accuracies')
    parser.add_argument('--MY_DETECTOR.C_NUM', type=int, default=None, help='C num')
    parser.add_argument('--MY_DETECTOR.DELTA', type=float, default=None, help='Delta')
    parser.add_argument('--MY_DETECTOR.TEMP', type=float, default=None, help='Temperature')
    parser.add_argument('--MY_DETECTOR.UC_MECH', type=str, default=None, help='UC mechanism')
    parser.add_argument('--MY_DETECTOR.SIGNIFICANCE_LEVEL', type=float, default=None, help='Significance level')
    parser.add_argument('--IMAGENET.DEVICE_INDEX', type=int, default=None, help='Device index')
    parser.add_argument('--IMAGENET.NUM_WORKERS', type=int, default=None, help='Number of workers')

    parser.add_argument('--ablation', action='store_true', default=False, help='whether it is an ablation of mine or not')


    args = parser.parse_args()
    return args


def override_cfg_with_args_from_command_line(args, config):
    # Update configuration file with command line arguments
    if args['IMAGENET.GET_ACCURACY_ONLY'] is not None:
        config['IMAGENET']['GET_ACCURACY_ONLY'] = args['IMAGENET.GET_ACCURACY_ONLY']
    if args['IMAGENET.DATASETS.PATH_TO_IMAGENET'] is not None:
        config['IMAGENET']['DATASETS']['PATH_TO_IMAGENET'] = args['IMAGENET.DATASETS.PATH_TO_IMAGENET']
    if args['IMAGENET.DATASETS.PATH_TO_IMAGENET_O'] is not None:
        config['IMAGENET']['DATASETS']['PATH_TO_IMAGENET_O'] = args['IMAGENET.DATASETS.PATH_TO_IMAGENET_O']
    if args['IMAGENET.DATASETS.PATH_TO_IMAGENET_A'] is not None:
        config['IMAGENET']['DATASETS']['PATH_TO_IMAGENET_A'] = args['IMAGENET.DATASETS.PATH_TO_IMAGENET_A']
    if args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['FGSM']['EPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['FGSM'][
            'BATCH_SIZE'] = args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.C'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['C'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.C']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['KAPPA'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['STEPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.LR'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['LR'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.LR']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD']['EPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD'][
            'STEPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD'][
            'ALPHA'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD'][
            'RANDOM_START'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START']
    if args['IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['GAUSS_NOISE'][
            'STD'] = args['IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD']
    if args['IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['ROTATE'][
            'ANGLE'] = args['IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE']
    if args['IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['ZOOM'][
            'FACTOR'] = args['IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR']
    if args['IMAGENET.EXPERIMENT.SEED'] is not None:
        config['IMAGENET']['EXPERIMENT']['SEED'] = args['IMAGENET.EXPERIMENT.SEED']
    if args['IMAGENET.EXPERIMENT.OOD'] is not None:
        config['IMAGENET']['EXPERIMENT']['OOD'] = args['IMAGENET.EXPERIMENT.OOD']
    if args['IMAGENET.EXPERIMENT.MMD_FIT_SIZE'] is not None:
        config['IMAGENET']['EXPERIMENT']['MMD_FIT_SIZE'] = args['IMAGENET.EXPERIMENT.MMD_FIT_SIZE']
    if args['IMAGENET.EXPERIMENT.KS_FIT_SIZE'] is not None:
        config['IMAGENET']['EXPERIMENT']['KS_FIT_SIZE'] = args['IMAGENET.EXPERIMENT.KS_FIT_SIZE']
    if args['IMAGENET.EXPERIMENT.TRAIN_SIZE'] is not None:
        config['IMAGENET']['EXPERIMENT']['TRAIN_SIZE'] = args['IMAGENET.EXPERIMENT.TRAIN_SIZE']
    if args['IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS'] is not None:
        config['IMAGENET']['EXPERIMENT']['NUM_RANDOM_RUNS'] = args['IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS']
    if args['IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS'] is not None:
        config['IMAGENET']['EXPERIMENT']['NUM_BOOTSTRAP_RUNS'] = args['IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS']
    if args['IMAGENET.EXPERIMENT.WINDOW_SIZES'] is not None:
        config['IMAGENET']['EXPERIMENT']['WINDOW_SIZES'] = args['IMAGENET.EXPERIMENT.WINDOW_SIZES']

    if args['IMAGENET.EXPERIMENT.DETECTORS'] is not None:
        config['IMAGENET']['EXPERIMENT']['DETECTORS'] = args['IMAGENET.EXPERIMENT.DETECTORS']

    if args['IMAGENET.EXPERIMENT.PATH_TO_RESULTS'] is not None:
        config['IMAGENET']['EXPERIMENT']['PATH_TO_RESULTS'] = args['IMAGENET.EXPERIMENT.PATH_TO_RESULTS']

    if args['IMAGENET.MODEL'] is not None:
        config['IMAGENET']['MODEL'] = args['IMAGENET.MODEL']

    if args['IMAGENET.PATH_TO_SAVE_OUTPUTS'] is not None:
        config['IMAGENET']['PATH_TO_SAVE_OUTPUTS'] = args['IMAGENET.PATH_TO_SAVE_OUTPUTS']

    if args['IMAGENET.BATCH_SIZE'] is not None:
        config['IMAGENET']['BATCH_SIZE'] = args['IMAGENET.BATCH_SIZE']

    if args['IMAGENET.PATH_TO_SAVE_ACCURACIES'] is not None:
        config['IMAGENET']['PATH_TO_SAVE_ACCURACIES'] = args['IMAGENET.PATH_TO_SAVE_ACCURACIES']

    if args['MY_DETECTOR.C_NUM'] is not None:
        config['MY_DETECTOR']['C_NUM'] = args['MY_DETECTOR.C_NUM']

    if args['MY_DETECTOR.DELTA'] is not None:
        config['MY_DETECTOR']['DELTA'] = args['MY_DETECTOR.DELTA']

    if args['MY_DETECTOR.TEMP'] is not None:
        config['MY_DETECTOR']['TEMP'] = args['MY_DETECTOR.TEMP']

    if args['MY_DETECTOR.UC_MECH'] is not None:
        config['MY_DETECTOR']['UC_MECH'] = args['MY_DETECTOR.UC_MECH']

    if args['MY_DETECTOR.SIGNIFICANCE_LEVEL'] is not None:
        config['MY_DETECTOR']['SIGNIFICANCE_LEVEL'] = args['MY_DETECTOR.SIGNIFICANCE_LEVEL']

    if args['IMAGENET.DEVICE_INDEX'] is not None:
        config['IMAGENET']['DEVICE_INDEX'] = args['IMAGENET.DEVICE_INDEX']
    if args['IMAGENET.NUM_WORKERS'] is not None:
        config['IMAGENET']['NUM_WORKERS'] = args['IMAGENET.NUM_WORKERS']


    return config



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

def calculate_averages(parent_path, file_name, new_file_name, model):
    # Get a list of all child directories
    child_dirs = [os.path.join(parent_path, child) for child in os.listdir(parent_path) if
                  os.path.isdir(os.path.join(parent_path, child))]

    # Create an empty list to hold the data from all files
    child_dirs = [s for s in child_dirs if model not in s]
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


# ====================================================================================== #
#                                 Dictionary handler                                     #
# ====================================================================================== #

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

# ====================================================================================== #
#                                  Results saver                                         #
# ====================================================================================== #

def save_data_my_detector_imagenet(cfg, result_dicts, window_size, threshold=0.95, suffix='', shift_extra_params=''):
    plot_lower_bounds(cfg, should_hold=True, window_size=window_size,
                 lower_bounds=get_values_from_list_dict(result_dicts, ['in_vs_in', str(window_size), 'my_detector',
                                                                       'coverage_lower_bounds']),
                 actual_coverage_for_lower=get_values_from_list_dict(result_dicts,
                                                                     ['in_vs_in', str(window_size), 'my_detector',
                                                                      'actual_coverages_lower']),
                 desired_coverages=get_values_from_list_dict(result_dicts,
                                                             ['in_vs_in', str(window_size), 'my_detector',
                                                              'desired_coverages'])[0], suffix=suffix,
                 shift_extra_params=shift_extra_params)

    plot_lower_bounds(cfg, should_hold=False, window_size=window_size,
                 lower_bounds=get_values_from_list_dict(result_dicts, ['in_vs_out', str(window_size), 'my_detector',
                                                                       'coverage_lower_bounds']),
                 actual_coverage_for_lower=get_values_from_list_dict(result_dicts,
                                                                     ['in_vs_out', str(window_size), 'my_detector',
                                                                      'actual_coverages_lower']),
                 desired_coverages=get_values_from_list_dict(result_dicts,
                                                             ['in_vs_out', str(window_size), 'my_detector',
                                                              'desired_coverages'])[0], suffix=suffix,
                 shift_extra_params=shift_extra_params)

    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'my_detector', 'final_score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'my_detector', 'final_score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/my_detector_window_{window_size}' + suffix + f'.csv'
    # print(csv_path)
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_ks_embs_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_ks_embs', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_ks_embs', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_ks_embs_window_{window_size}' + f'.csv'
    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_ks_logits_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_ks_logits', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_ks_logits', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)

    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_ks_logits_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_ks_softmaxes_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_ks_softmaxes', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_ks_softmaxes', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_ks_softmaxes_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_mmd_embs_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_mmd_embs', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_mmd_embs', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_mmd_embs_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_mmd_logits_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_mmd_logits', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_mmd_logits', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_mmd_logits_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_mmd_softmaxes_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_mmd_softmaxes', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_mmd_softmaxes', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_mmd_softmaxes_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_single_softmaxes_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_single_softmaxes', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_single_softmaxes', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_single_softmaxes_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_single_entropies_imagenet(cfg, result_dicts, window_size, threshold=0.95, shift_extra_params=''):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_single_entropies', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_single_entropies', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_single_entropies_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


