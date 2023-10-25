import numpy as np
from scipy.stats import binom
import torch
import tqdm
import sys
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from utils import *
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from torch.distributions import Categorical


def get_softmax_responses(logits_list, temperature=1.0):
    """
    Args:
        softmax_list (list): List of PyTorch tensors or NumPy arrays, each containing a vector of softmax.
        temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.

    Returns:
        List of PyTorch tensors or NumPy arrays, each containing a vector of softmax responses corresponding to the input logits.
    """
    # max_values = np.amax(softmax_list, axis=1)
    # return max_values
    if isinstance(logits_list[0], np.ndarray):
        logits_list = np.stack(logits_list)

    # softmax = softmax(logits_list, temperature)
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


class SGC():
    def __init__(self, S_m, delta, c_star, bound, TOL=1e-10):
        """
        Initializes the SGC algorithm with the given parameters.
        Example usage:

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


class Shift_Detector():

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

    def detect(self, us_out_dist):

        def get_violations_lower(nums, arrays):
                        # Convert input to numpy arrays
            nums = np.array(nums)
            arrays = np.array(arrays)

            # Compute the mean of each row in arrays
            mean_vals = arrays.mean(axis=1)

            # Compute the mask based on the condition
            mask = (nums > mean_vals).astype(int)

            # Set arrays rows to zeros where the condition didn't hold
            arrays[mask == 0] = 0

            return arrays.tolist(), mask.tolist()
            # mask = []
            # for i in range(len(nums)):
            #     if nums[i] > sum(arrays[i]) / len(arrays[i]):  # didn't hold!
            #         mask.append(1)  # didn't hold!
            #     else:
            #         arrays[i] = [0] * len(arrays[i])
            #         mask.append(0)  # hold!
            # return arrays, mask

        us_out_dist_np = np.array(us_out_dist)
        thresholds_np = np.array(self.Thresholds_For_Lower_Bounds)
        # Broadcasting to create a 2D array of shape (len(thresholds), len(us_out_dist))
        zero_one_coverages_lower_np = (us_out_dist_np > thresholds_np[:, None]).astype(int)
        zero_one_coverages_lower = zero_one_coverages_lower_np.tolist()

        self.Actual_coverage_for_Lower = [sum(sublist) / len(sublist) for sublist in zero_one_coverages_lower]

        # modify zero_one_coverages_lower and get violations_lower
        zero_one_coverages_lower_modified, mask_lower = get_violations_lower(self.Coverage_Lower_Bounds,
                                                                             zero_one_coverages_lower)
        violations_lower = [x for x, flag in zip(self.Coverage_Lower_Bounds, mask_lower) if flag] # will keep only violated thresholds
        sum_violations_lower = sum(violations_lower)
        # calculate final_list_lower
        final_list_lower = [sum(sublist) for sublist in zip(*zero_one_coverages_lower_modified)] # sums across samples
        
        final_list_lower = [-(x - sum_violations_lower) for x in final_list_lower]


        final_list = final_list_lower
        if max(final_list) == 0 and min(final_list) == 0:  # no violations!
            p_value = 1
        else:
            t_stat, p_value = ttest_1samp(final_list, 0, alternative='greater')
            # print(f'{p_value=}')
        return p_value
    
    def get_actual_coverage_for_lower(self):
        return self.Actual_coverage_for_Lower
    
    def get_lower_bounds(self):
        return self.Coverage_Lower_Bounds

    def get_desired_coverages(self):
        return self.c_star_arr

