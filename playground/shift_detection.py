
import torch
import numpy as np
from scipy.stats import binom, ttest_1samp
import matplotlib.pyplot as plt
import tqdm
from timeit import default_timer as timer
import os
import sys
import random
import time
from easydict import EasyDict as edict
from abc import ABC, abstractmethod
import torch.nn.functional as F
import pprint
from torch.distributions import Categorical


# ======================================== kappas ======================================== #
def get_softmax_responses(logits_list, temperature=1.0):
    """
    Args:
        softmax_list (list): List of PyTorch tensors or NumPy arrays, each containing a vector of softmax.
        temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.

    Returns:
        List of PyTorch tensors or NumPy arrays, each containing a vector of softmax responses corresponding to the input logits.
    """

    if isinstance(logits_list[0], np.ndarray):
        logits_list = np.stack(logits_list)

    softmax = F.softmax(logits_list / temperature, dim=-1)
    SR_list = torch.max(softmax, dim=1)[0].tolist()

    return SR_list


def get_entropy_of_softmax(logits_list, temperature=1.0):
    """
    Args:
        logits_list (list): List of PyTorch tensors or NumPy arrays, each containing a vector of logits.
        temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.

    Returns:
        List of entropy values for each softmax vector.
    """
    if isinstance(logits_list[0], np.ndarray):
        logits_list = np.stack(logits_list)

    # Convert to PyTorch tensor and apply softmax with temperature
    # logits_tensor = torch.tensor(logits_list) / temperature
    logits_tensor = logits_list / temperature
    softmax = F.softmax(logits_tensor, dim=-1)

    # Compute entropy of each softmax vector
    dist = Categorical(probs=softmax)
    entropy = dist.entropy()

    # Normalize entropy to [0, 1] range
    entropy = entropy / np.log(logits_tensor.shape[1])
    # entropy = np.clip(entropy.numpy(), 0, 1)
    kappa = 1 - entropy

    return kappa.tolist()
# ======================================================================================== #


# ======================================== helpers ======================================== #
def print_nested_dict(nested_dict):
    pprint.pprint(nested_dict)


def plot_bounds(upper_bounds, 
                lower_bounds, 
                actual_coverage_for_lower,
                actual_coverage_for_upper,
                desired_coverages):
    """
    This function plots the upper and lower bounds as well as the actual coverage for each coverage parameter. The desired
    coverages are also plotted as scatter points. Violations of the upper and lower bounds are marked as red and blue
    circles, respectively.

    Args:
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
    actual_coverage_for_upper_std = np.std(actual_coverage_for_upper, axis=0) / np.sqrt(
        len(actual_coverage_for_upper))
    actual_coverage_for_lower_mean = np.mean(actual_coverage_for_lower, axis=0)
    actual_coverage_for_lower_std = np.std(actual_coverage_for_lower, axis=0) / np.sqrt(
        len(actual_coverage_for_upper))

    upper_bounds = np.mean(upper_bounds, axis=0)
    lower_bounds = np.mean(lower_bounds, axis=0)
    # TODO: uncomment this if you want to plot upper bounds
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

    # TODO: uncomment this if you want to plot upper bounds
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
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


    # Show the plot
    plt.show()

def print_p_value_info(p_value, alpha=0.05, bound='lower'):
    """
    Prints the p-value and indicates if a shift is detected based on the given confidence parameter.
    
    Parameters:
    - p_value (float): The p-value to check.
    - alpha (float): The confidence parameter. Default is 0.05.
    - bound (str): Specifies if it's a lower or upper bound test. Accepted values are 'lower' or 'upper'.
    
    Returns:
    None
    """
    if bound == 'lower':
        bound = 'under'
    elif bound == 'upper':
        bound = 'over'
    else:
        raise ValueError("bad bound argument!")
        
    
    # Format the header based on the bound
    header = f"{'#' * 5} {bound.capitalize()} Confidence Shift {'#' * 5}"
    
    # Format the result based on the p_value
    result = "Shift detected!" if p_value < alpha else "No shift detected."
    
    print(header)
    print(f"P-value: {p_value:.4f}")
    print(result)
    print('#' * len(header))


# ======================================================================================== #

# ========== SGC, SGC-UP and bounds fitter (SGC several times) =========================== #
class SGC():
    def __init__(self, S_m, delta, c_star, bound, TOL=1e-10):
        """
        Initializes the SGC algorithm with the given parameters.

        Args:
            S_m (list or torch.Tensor): The m i.i.d samples - assuming we get a vector of kappa values of the instances,
                for example: [softmax_response,softmax_response,...]
                We run the algo on it to get values of [theta_i, c_i].
            delta (float): The confidence parameter for the coverage.
            c_star (int): The desired coverage.
            bound (str): The upper bound or lower bound - 'U' or 'L'.
            TOL (float): The tolerance for the convergence.
        """
        self.TOL = TOL
        self.delta = delta
        self.c_star = c_star
        self.m = len(S_m)
        if bound not in ['U', 'L']:
            raise ValueError("Bound should be 'U' or 'L'.")
        self.bound = bound

        if isinstance(S_m, list) or isinstance(S_m, np.ndarray):
            S_m = torch.tensor(S_m)
        self.S_m = S_m

    def loop(self):
        """
        Runs the SGC algorithm.

        Returns:
            A tuple containing the guaranteed coverage and the corresponding theta value.
        """
        if self.bound == 'L':
            return self._loop_L()
        else:
            return self._loop_U()

    def _loop_L(self):
        """
        Runs the SGC algorithm with the lower bound.

        Returns:
            A tuple containing the guaranteed coverage and the corresponding theta value.
        """

        z_min = 0
        z_max = self.m - 1
        k = int(np.ceil(np.log2(self.m) + 1)) + 1
        sorted_S_m = self.S_m_sorter()
        for i in range(k):
            z = np.ceil((z_min + z_max) / 2)
            theta_z = sorted_S_m[int(z)]
            c_hat_z = int((self.m - z))
            guaranteed_c = self.bin_tale_L(self.m, c_hat_z, 1 - self.delta / k)
            if guaranteed_c < self.c_star:
                z_max = z
            else:
                z_min = z

        # print(f'The guaranteed coverage is: {guaranteed_c} and the corresponding threshold is {theta_z.item()}')
        return guaranteed_c, theta_z

    def _loop_U(self):
        """
        Runs the SGC algorithm with the upper bound.

        Returns:
            A tuple containing the guaranteed coverage and the corresponding theta value.
        """
        z_min = 0
        z_max = self.m - 1
        k = int(np.ceil(np.log2(self.m + 1))) + 1
        sorted_S_m = self.S_m_sorter()
        for i in range(k):
            z = np.floor((z_min + z_max) / 2)
            theta_z = sorted_S_m[int(z)]
            c_hat_z = int((self.m - z))
            guaranteed_c = self.bin_tale_U(self.m, c_hat_z, self.delta / k)
            if guaranteed_c > self.c_star:
                z_min = z
            else:
                z_max = z

        # print(f'The guaranteed coverage is: {guaranteed_c} and the corresponding threshold is {theta_z.item()}')
        return guaranteed_c, theta_z

    def bin_tale_L(self, m, c_hat, delta):
        """
        Computes the probability p such that Bin(m, c_hat, p) = delta.

        Args:
            m (int): The number of samples.
            c_hat (int): The number of successes.
            delta (float): The desired probability.
        Returns:
            float: The probability p such that Bin(m, c_hat, p) = delta.

        Raises:
            ValueError: If `delta` is not a valid probability or if `c_hat` is greater than `m`.
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be a valid probability between 0 and 1.")
        if c_hat > m:
            raise ValueError("c_hat must be less than or equal to m.")
        p_1 = 0
        p_2 = c_hat / m

        p = (p_1 + p_2) / 2
        Bin = binom.cdf(c_hat, m, p)
        while (abs(delta - Bin) > self.TOL):

            if (Bin > delta):
                p_1 = p
            elif (Bin < delta):
                p_2 = p
            p = (p_1 + p_2) / 2
            Bin = binom.cdf(c_hat, m, p)
        return p

    def bin_tale_U(self, m, c_hat, delta):
        """
        Computes the probability p such that Bin(m, c_hat, p) = delta.

        Args:
            m (int): The number of samples.
            c_hat (int): The number of successes.
            delta (float): The desired probability.
        Returns:
            float: The probability p such that Bin(m, c_hat, p) = delta.

        Raises:
            ValueError: If `delta` is not a valid probability or if `c_hat` is greater than `m`.
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be a valid probability between 0 and 1.")
        if c_hat > m:
            raise ValueError("c_hat must be less than or equal to m.")
        p_1 = c_hat / m
        p_2 = 1

        p = (p_1 + p_2) / 2
        Bin = binom.cdf(c_hat, m, p)
        while (abs(delta - Bin) > self.TOL):
            if (Bin > delta):
                p_1 = p
            elif (Bin < delta):
                p_2 = p
            p = (p_1 + p_2) / 2
            Bin = binom.cdf(c_hat, m, p)
        return p

    def S_m_sorter(self):
        """

        Returns: a sorted list of S_m

        """
        sorted_S_m, _ = torch.sort(self.S_m)
        return sorted_S_m

class Bounds_fitter():

    def __init__(self, C_num, delta):
        """
        Initialize the class with uncertainty estimators and parameters.


        Args:
            C_num (int): The number of different coverage values to bound. Note that C_num has to be big if way == second.
            delta (float): The confidence parameter for the algorithm.

        Attributes:
            C_num (int): The number of different coverage values to bound.
            delta (float): The confidence parameter for the algorithm.
            S_m_tot_size (int): The length of the array of uncertainty estimators for the window.

        """

        self.C_num = C_num
        self.delta = delta

        self.c_star_arr = np.arange(0.1, 1, 1 / self.C_num)

    def fit_lower_bound(self, us_in_dist):
        """
        Args:
            us_in_dist (array-like): An array of uncertainty estimators for the in-distribution data.

        Runs the test and creates the following attributes:
            Coverage_Lower_Bounds: A list of lower bounds.
            Thresholds_For_Lower_Bounds: A list of thresholds for lower bounds.

         """
        self.S_n_us = us_in_dist

        num_iterations = len(self.c_star_arr)

        self.Coverage_Lower_Bounds = []
        self.Thresholds_For_Lower_Bounds = []

        with tqdm.tqdm(desc="Fitting for lower bound", total=num_iterations, file=sys.stdout) as pbar:
            timer_start = timer()
            # This is a single iteration
            for c in self.c_star_arr:
                # Initializing the algorithm
                algorithm = SGC(self.S_n_us, self.delta, c, 'L')
                # Gets the coverage bound, theta
                coverage_lower_bound, threshold_for_lower_bound = algorithm.loop()

                # Appending to the arrays of the coverage bound and thresholds
                self.Coverage_Lower_Bounds.append(coverage_lower_bound)
                self.Thresholds_For_Lower_Bounds.append(threshold_for_lower_bound)

                pbar.set_description(f'Fitting for lower bound, Elapsed time: {timer() - timer_start:.3f} sec')
                pbar.update()

    def fit_upper_bound(self, us_in_dist):
        """
         Runs the test and creates the following attributes:
             Coverage_Upper_Bounds: A list of upper bounds.
             Thresholds_For_Upper_Bounds: A list of thresholds for upper bounds.
             Actual_coverage_for_Upper: Actual coverage calculated with thresholds of upper bounds.
             bad_events_percentage_for_upper: The percentage that the upper bound didn't hold.
         """
        self.S_n_us = us_in_dist

        num_iterations = len(self.c_star_arr)
        self.Coverage_Upper_Bounds = []
        self.Thresholds_For_Upper_Bounds = []

        with tqdm.tqdm(desc="Fitting for upper bound", total=num_iterations, file=sys.stdout) as pbar:
            timer_start = timer()
            # This is a single iteration
            for c in self.c_star_arr:
                # Initializing the algorithm
                algorithm = SGC(self.S_n_us, self.delta, c, 'U')
                # Gets the coverage bound, theta
                coverage_upper_bound, threshold_for_upper_bound = algorithm.loop()

                # Appending to the arrays of the coverage bound and thresholds
                self.Coverage_Upper_Bounds.append(coverage_upper_bound)
                self.Thresholds_For_Upper_Bounds.append(threshold_for_upper_bound)

                pbar.set_description(f'Fitting for upper bound, Elapsed time: {timer() - timer_start:.3f} sec')
                pbar.update()

    def detect(self, us_out_dist):

        def get_violations_lower(nums, arrays):
            mask = []
            for i in range(len(nums)):
                if nums[i] > sum(arrays[i]) / len(arrays[i]): 
                    mask.append(1) 
                else:
                    arrays[i] = [0] * len(arrays[i])
                    mask.append(0) 
            return arrays, mask

        def get_violations_upper(nums, arrays):
            mask = []
            for i in range(len(nums)):
                if nums[i] < sum(arrays[i]) / len(arrays[i]):
                    mask.append(1) 
                else:
                    arrays[i] = [0] * len(arrays[i])
                    mask.append(0) 
            return arrays, mask

        # lower bound p-value
        zero_one_coverages_lower = [[int(x > threshold) for x in us_out_dist] for threshold in
                                    self.Thresholds_For_Lower_Bounds]
        self.Actual_coverage_for_Lower = [sum(sublist) / len(sublist) for sublist in zero_one_coverages_lower]

        zero_one_coverages_lower_modified, mask_lower = get_violations_lower(self.Coverage_Lower_Bounds,
                                                                             zero_one_coverages_lower)
        violations_lower = [x for x, flag in zip(self.Coverage_Lower_Bounds, mask_lower) if flag]


        final_list_lower = [sum(sublist) for sublist in zip(*zero_one_coverages_lower_modified)]
        final_list_lower = [x - sum(violations_lower) for x in final_list_lower]
        final_list_lower = [-x for x in final_list_lower]
        if max(final_list_lower) == 0 and min(final_list_lower) == 0:  # no violations!
            p_value_lower = 1
        else:
            t_stat_lower, p_value_lower = ttest_1samp(final_list_lower, 0, alternative='greater')

        # upper bound p-value
        
        zero_one_coverages_upper = [[int(x > threshold) for x in us_out_dist] for threshold in
                                    self.Thresholds_For_Upper_Bounds]
        self.Actual_coverage_for_Upper = [sum(sublist) / len(sublist) for sublist in zero_one_coverages_upper]

        zero_one_coverages_upper_modified, mask_upper = get_violations_upper(self.Coverage_Upper_Bounds,
                                                                             zero_one_coverages_upper)
        
        violations_upper = [x for x, flag in zip(self.Coverage_Upper_Bounds, mask_upper) if flag]

        final_list_upper = [sum(sublist) for sublist in zip(*zero_one_coverages_upper_modified)]
        final_list_upper = [sum(violations_upper) - x for x in final_list_upper]
        final_list_upper = [-x for x in final_list_upper]
        if max(final_list_upper) == 0 and min(final_list_upper) == 0:  # no violations!
            p_value_upper = 1
        else:
            t_stat_upper, p_value_upper = ttest_1samp(final_list_upper, 0, alternative='greater')
        return p_value_lower, p_value_upper

    # ========================================================

    def detect_lower_bound_deviation(self, us_window, return_p_value=False):
        """
        detects deviation between actual coverage and expected coverage for a given lower bound.

        Parameters:
        -----------
        us_window: list
            A list of values to use as the upper bound for each iteration.
        return_p_value: Boolean
            A Flag indicating whether to return the p-value or not.

        Returns:
        --------
        float
            A score representing the degree of under confidence in the estimates, or the p-value if return_p_value is True.
        """
        self.S_m_us = us_window
        self.S_m_tot_size = len(self.S_m_us)
        self.Actual_coverage_for_Lower = []
        self.bad_events_percentage_for_lower = 0
        num_iterations = len(self.Coverage_Lower_Bounds)
        with tqdm.tqdm(desc="Testing for lower bound deviation", total=num_iterations, file=sys.stdout) as pbar:
            timer_start = timer()
            # This is a single iteration
            for coverage_lower_bound, threshold_for_lower_bound in zip(self.Coverage_Lower_Bounds,
                                                                       self.Thresholds_For_Lower_Bounds):

                # Calculating the actual coverage
                actual_coverage_for_lower_bound = self._calculate_coverage_on_valid(threshold_for_lower_bound)
                # Checking if we got a bad event
                if actual_coverage_for_lower_bound < coverage_lower_bound:
                    self.bad_events_percentage_for_lower = self.bad_events_percentage_for_lower + 1

                # appending to actual coverage array
                self.Actual_coverage_for_Lower.append(actual_coverage_for_lower_bound)

                pbar.set_description(
                    f'Testing for lower bound deviation, Elapsed time: {timer() - timer_start:.3f} sec')
                pbar.update()
        return self._get_under_confidence_score(return_p_value)

    def get_lower_bounds(self):
        return self.Coverage_Lower_Bounds

    def get_actual_coverage_for_lower(self):
        return self.Actual_coverage_for_Lower

    def get_actual_coverage_for_upper(self):
        return self.Actual_coverage_for_Upper

    def get_upper_bounds(self):
        return self.Coverage_Upper_Bounds

    def get_desired_coverages(self):
        return self.c_star_arr

    # =================================================================

# ====================================== detector ======================================== #
class Detector(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def detect(self, X):
        pass
    
class CBD(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # handle any additional arguments as needed
        self.c_num = kwargs.get('c_num')
        self.delta = kwargs.get('delta')
        self.temp = kwargs.get('temprature', 1.0)
        self.uncertainty_mechanism = kwargs.get('uncertainty_mechanism')
        self.detector = Bounds_fitter(self.c_num, self.delta)

    def fit(self, X):
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # train the detector on X and y
        if self.uncertainty_mechanism == 'Ent':
            self.kappa_extractor = get_entropy_of_softmax
        elif self.uncertainty_mechanism == 'SR':
            self.kappa_extractor = get_softmax_responses
        else:
            raise Exception("No such uncertainty mechanism")
            exit()
        X = self.kappa_extractor(X, self.temp)

        # sys.stdout = open(os.devnull, 'w')
        self.detector.fit_lower_bound(X)
        self.detector.fit_upper_bound(X)

        self.desired_coverages = self.detector.get_desired_coverages()
        self.coverage_lower_bounds = self.detector.get_lower_bounds()
        self.coverage_upper_bounds = self.detector.get_upper_bounds()
        sys.stdout = stdout

    def detect(self, X):
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        # apply the detector to X and return the results
        X = self.kappa_extractor(X, self.temp)
        start_time = time.time()
        self.p_value_lower, self.p_value_upper = self.detector.detect(X)
        end_time = time.time()
        self.actual_coverages_lower = self.detector.get_actual_coverage_for_lower()
        self.actual_coverages_upper = self.detector.get_actual_coverage_for_upper()
        self.detection_time = end_time - start_time
        self.detection_parameters = edict({
            'desired_coverages': self.desired_coverages,
            'coverage_lower_bounds': self.coverage_lower_bounds,
            'coverage_upper_bounds': self.coverage_upper_bounds,

            'detection_time': self.detection_time,
            'actual_coverages_lower': self.actual_coverages_lower,

            'actual_coverages_upper': self.actual_coverages_upper,

            'p_value_lower': self.p_value_lower,
            'p_value_upper': self.p_value_upper,

        })
        sys.stdout = stdout
        return self.detection_parameters.copy()
 # ======================================================================================== #