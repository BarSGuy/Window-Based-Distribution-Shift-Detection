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


# def get_softmax_responses(logits_list, temperature=1.0):
#     """
#     Args:
#         logits_list (list): List of PyTorch tensors or NumPy arrays, each containing a vector of logits.
#         temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
#
#     Returns:
#         List of PyTorch tensors or NumPy arrays, each containing a vector of softmax responses corresponding to the input logits.
#     """
#     SR_list = []
#     for logits in logits_list:
#         if isinstance(logits, np.ndarray):
#             logits = torch.from_numpy(logits)
#         softmax = F.softmax(logits / temperature, dim=-1)
#         SR = torch.max(softmax).item()
#         SR_list.append(SR)
#     return SR_list


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

            S_m = sorted(generate_random_numbers_between_0_and_1(1000))
            delta = 0.001
            c_star = 0.5
            bound = 'L'
            sgc = SGC(S_m, delta, c_star, bound)
            guaranteed_coverage, corresponding_theta= sgc.loop()



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
        Example usage:

            us_in_dist = sorted(generate_random_numbers_between_0_and_1(1000))
            us_window = sorted(generate_random_numbers_between_0_and_1(1000))
            C_num = 10
            delta = 0.0001

            detector = Shift_Detector(C_num, delta)

            detector.fit_lower_bound(us_in_dist)
            under_confidence_score = detector.detect_lower_bound_deviation(us_window)
            detector.visualize_lower_bound()

            detector.fit_upper_bound(us_in_dist)
            over_confidence_score = detector.detect_upper_bound_deviation(us_window)
            detector.visualize_upper_bound()


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
                if nums[i] > sum(arrays[i]) / len(arrays[i]):  # didn't hold!
                    mask.append(1)  # didn't hold!
                else:
                    arrays[i] = [0] * len(arrays[i])
                    mask.append(0)  # hold!
            return arrays, mask

        def get_violations_upper(nums, arrays):
            mask = []
            for i in range(len(nums)):
                if nums[i] < sum(arrays[i]) / len(arrays[i]):  # didn't hold!
                    mask.append(1)  # didn't hold!
                else:
                    arrays[i] = [0] * len(arrays[i])
                    mask.append(0)  # hold!
            return arrays, mask

        # get zero_one_coverages_lower
        zero_one_coverages_lower = [[int(x > threshold) for x in us_out_dist] for threshold in
                                    self.Thresholds_For_Lower_Bounds]
        self.Actual_coverage_for_Lower = [sum(sublist) / len(sublist) for sublist in zero_one_coverages_lower]
        # modify zero_one_coverages_lower and get violations_lower
        zero_one_coverages_lower_modified, mask_lower = get_violations_lower(self.Coverage_Lower_Bounds,
                                                                             zero_one_coverages_lower)
        violations_lower = [x for x, flag in zip(self.Coverage_Lower_Bounds, mask_lower) if flag] # will keep only violated thresholds

        # calculate final_list_lower
        final_list_lower = [sum(sublist) for sublist in zip(*zero_one_coverages_lower_modified)] # sums across samples
        final_list_lower = [x - sum(violations_lower) for x in final_list_lower]
        final_list_lower = [-x for x in final_list_lower]

        # # get actual coverages lower
        # zero_one_coverages_lower = []
        # for threshold in self.Thresholds_For_Lower_Bounds:
        #     zero_one_coverages_lower.append([1 if x > threshold else 0 for x in us_out_dist])
        # zero_one_coverages_lower_modified, mask_lower = modify_arrays_lower(self.Coverage_Lower_Bounds,
        #                                                                     zero_one_coverages_lower)
        # violations_lower = [x for x, flag in zip(self.Coverage_Lower_Bounds, mask_lower) if flag]
        # final_list_lower = [sum(sublist[i] for sublist in zero_one_coverages_lower_modified) for i in range(len(zero_one_coverages_lower_modified[0]))]
        # final_list_lower1 = [final_list_lower[i] - sum(violations_lower) for i in range(len(final_list_lower))]
        # final_list_lower1 = [-x for x in final_list_lower1]

        # get actual coverages upper
        # zero_one_coverages_upper = []
        # for threshold in self.Thresholds_For_Upper_Bounds:
        #     zero_one_coverages_upper.append([1 if x > threshold else 0 for x in us_out_dist])
        #
        # zero_one_coverages_upper_modified, mask_upper = get_violations_upper(self.Coverage_Upper_Bounds,
        #                                                                     zero_one_coverages_upper)
        # violations_upper = [x for x, flag in zip(self.Coverage_Upper_Bounds, mask_upper) if flag]
        # final_list_upper = [sum(sublist[i] for sublist in zero_one_coverages_upper_modified) for i in
        #                     range(len(zero_one_coverages_upper_modified[0]))]
        # final_list_upper1 = [sum(violations_upper) - final_list_upper[i] for i in range(len(final_list_upper))]

        ## get zero_one_coverages_upper
        zero_one_coverages_upper = [[int(x > threshold) for x in us_out_dist] for threshold in
                                    self.Thresholds_For_Upper_Bounds]
        self.Actual_coverage_for_Upper = [sum(sublist) / len(sublist) for sublist in zero_one_coverages_upper]
        # # modify zero_one_coverages_upper and get violations_upper
        # zero_one_coverages_upper_modified, mask_upper = get_violations_upper(self.Coverage_Upper_Bounds,
        #                                                                      zero_one_coverages_upper)
        # violations_upper = [x for x, flag in zip(self.Coverage_Upper_Bounds, mask_upper) if flag] # keeps only violated coverages
        #
        # # calculate final_list_upper
        # final_list_upper = [sum(sublist) for sublist in zip(*zero_one_coverages_upper_modified)]
        # final_list_upper = [sum(violations_upper) - x for x in final_list_upper]
        # # final_list_upper = [x - sum(violations_upper) for x in final_list_upper]


        # final_list_old = [a + b for a, b in zip(final_list_upper, final_list_lower)]

        # TODO: if you want the last working version, use p_value_old as the returned p-value and uncomment
        # TODO: whats under get zero_one_coverages_upper, and the line with the commented TODO after it
        final_list = final_list_lower
        if max(final_list) == 0 and min(final_list) == 0:  # no violations!
            p_value = 1
        else:
            t_stat, p_value = ttest_1samp(final_list, 0, alternative='greater')
            # t_stat, p_value_old = ttest_1samp(final_list_old, 0, alternative='greater') # TODO
            # print(f'{p_value=}_{p_value_old=}')
        return p_value

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

    def visualize_lower_bound(self, title='Lower Bound', save_path="./figures"):
        """
        Args:
            title: The title of the graph
            save_path: The path to save the plot


        Visualizes the lower bound:
        Plots two graphs of actual bounds vs actual coverage values and difference
        """

        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        fig.tight_layout(pad=3.0, w_pad=4.0)

        fig.suptitle(title)

        # Plot actual coverage vs lower bound
        axs[0].plot(self.Coverage_Lower_Bounds, self.Actual_coverage_for_Lower, "b")
        axs[0].plot(self.Coverage_Lower_Bounds, self.Coverage_Lower_Bounds, "g--", label="($y=x$) Lower Bound")
        axs[0].legend(loc='best')
        axs[0].grid()

        axs[0].set_ylabel("Empirical Coverage")
        axs[0].set_xlabel('$c^*$')

        # Plot difference
        diff = torch.tensor(self.Actual_coverage_for_Lower) - torch.tensor(self.Coverage_Lower_Bounds)
        axs[1].plot(self.Coverage_Lower_Bounds, diff, "r")
        # axs[1].legend(loc='best')
        axs[1].grid()

        axs[1].set_ylabel("Gap")
        axs[1].set_xlabel('$c^*$')

        # Save plot
        plt.savefig(os.path.join(save_path, title + '.pdf'))
        plt.show()

    def detect_upper_bound_deviation(self, us_window, return_p_value=False):
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
        self.Actual_coverage_for_Upper = []
        self.bad_events_percentage_for_upper = 0
        num_iterations = len(self.Coverage_Upper_Bounds)
        with tqdm.tqdm(desc="Testing for upper bound deviation", total=num_iterations, file=sys.stdout) as pbar:
            timer_start = timer()
            # This is a single iteration
            for coverage_upper_bound, threshold_for_upper_bound in zip(self.Coverage_Upper_Bounds,
                                                                       self.Thresholds_For_Upper_Bounds):

                # Calculating the actual coverage
                actual_coverage_for_upper_bound = self._calculate_coverage_on_valid(threshold_for_upper_bound)
                # Checking if we got a bad event
                if actual_coverage_for_upper_bound < coverage_upper_bound:
                    self.bad_events_percentage_for_upper = self.bad_events_percentage_for_upper + 1

                # appending to actual coverage array
                self.Actual_coverage_for_Upper.append(actual_coverage_for_upper_bound)

                pbar.set_description(
                    f'Testing for upper bound deviation, Elapsed time: {timer() - timer_start:.3f} sec')
                pbar.update()
        return self._get_over_confidence_score(return_p_value)

    def visualize_upper_bound(self, title='Upper Bound', save_path="./figures"):
        """
        Args:
            title: The title of the graph
            save_path: The path to save the plot

        Visualizes the upper bound:
        Plots two graphs of actual bounds vs actual coverage values and difference
        """

        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        fig.tight_layout(pad=3.0, w_pad=4.0)

        fig.suptitle(title)

        # Plot actual coverage vs upper bound
        axs[0].plot(self.Coverage_Upper_Bounds, self.Actual_coverage_for_Upper, "b")
        axs[0].plot(self.Coverage_Upper_Bounds, self.Coverage_Upper_Bounds, "g--", label="($y=x$) Upper Bound")
        axs[0].legend(loc='best')
        axs[0].grid()

        axs[0].set_ylabel("Empirical Coverage")
        axs[0].set_xlabel('$c^*$')

        # Plot difference
        diff = torch.tensor(self.Actual_coverage_for_Upper) - torch.tensor(self.Coverage_Upper_Bounds)
        axs[1].plot(self.Coverage_Upper_Bounds, diff, "r")
        # axs[1].legend(loc='best')
        axs[1].grid()

        axs[1].set_ylabel("Gap")
        axs[1].set_xlabel('$c^*$')

        # Save plot
        plt.savefig(os.path.join(save_path, title + '.pdf'))
        plt.show()

    # =================================================================

    def _get_under_confidence_score(self, return_p_value):
        """
        Calculate the under-confidence score for the given data using hypothesis testing.

        args:
            return_p_value: a flag to indicate whether to return the p-value or not.

        Returns:
            The under-confidence score.
        """
        # Run a hypothesis test to calculate the p-value for under-confidence
        _, p_value_under = self._hypothesis_test('L')
        if return_p_value:
            return p_value_under
        # Calculate the under-confidence score as 1 minus the p-value
        self.phi_under = 1 - p_value_under
        # Return the under-confidence score
        return self.phi_under

    def _get_over_confidence_score(self, return_p_value):
        """
        Calculate the over-confidence score for the given data using hypothesis testing.

        args:
            return_p_value: a flag to indicate whether to return the p-value or not.

        Returns:
            The under-confidence score.
        """
        # Run a hypothesis test to calculate the p-value for over-confidence
        _, p_value_upper = self._hypothesis_test('U')
        if return_p_value:
            return p_value_upper
        # Calculate the over-confidence score as 1 minus the p-value
        self.phi_upper = 1 - p_value_upper
        # Return the under-confidence score
        return self.phi_upper

    def _calculate_coverage_on_valid(self, theta):
        """
        Calculates the actual coverage given the threshold theta and the bound on the coverage.

        Args:
            theta (torch.Tensor): The threshold value.
            bound (str): The bound on the coverage ('L' for lower bound or 'U' for upper bound).

        Returns:
            float: The actual coverage given the threshold theta.
        """
        # Convert S_m_us to a torch tensor if it is not already.
        if not torch.is_tensor(self.S_m_us):
            self.S_m_us = torch.tensor(self.S_m_us)
        # Get the indices where S_m_us is greater than or equal to theta.
        soft_max_vector_new = self.S_m_us[self.S_m_us >= theta.item()]
        # soft_max_vector_new = self.S_m_us[self.S_m_us > theta.item()]
        # Calculate the actual coverage
        coverage = len(soft_max_vector_new) / len(self.S_m_us)

        return coverage

    def find_largest_gap_index(self, list1, list2):
        """
        Find the index i of the largest gap between the values in two lists.

        Args:
            list1 (list): A list of numeric values.
            list2 (list): A list of numeric values with the same length as list1.

        Returns:
            int: The index i of the largest gap between the values in list1 and list2.
        """
        # Find the differences between corresponding elements in the two lists
        differences = [b - a for a, b in zip(list1, list2)]
        # Find the index of the maximum difference
        max_diff_index = differences.index(max(differences))
        return max_diff_index

    def _hypothesis_test(self, bound):
        """
        Runs a hypothesis test on the data based on a given bound.

        Args:
            bound (str): A string indicating the bound to use for the test ('L' for lower bound or 'U' for upper bound).

        Returns:
            A tuple containing the t-statistic and p-value from the hypothesis test.
        """
        self._get_new_x_j(bound)
        if bound == 'L':
            tail = 'less'
        else:
            tail = 'greater'
        from scipy import stats
        # Running Hypothesis test
        # H_0: both values are the same
        # H_1: the value of actual is {tail} than 0.5
        if bound == 'L':
            new_list_for_hyp = [sum(x) for x in zip(*self.array_of_Xjs_lower)]
            mean = sum(self.Coverage_Lower_Bounds)
            t_statistic, p_value = stats.ttest_1samp(a=new_list_for_hyp, popmean=mean, alternative=tail)
        else:
            mean = sum(self.Coverage_Upper_Bounds)
            new_list_for_hyp = [sum(x) for x in zip(*self.array_of_Xjs_upper)]
            t_statistic, p_value = stats.ttest_1samp(a=new_list_for_hyp,
                                                     popmean=mean, alternative=tail)
        return t_statistic, p_value

        # actual_coverages = [sum(x) / len(self.array_of_Xjs_lower[0]) for x in self.array_of_Xjs_lower]
        # gaps = [b - a for a, b in zip(actual_coverages, self.Coverage_Lower_Bounds)]  # should all be negative
        # gaps = [num if num >= 0 else 0 for num in gaps]
        # actual_coverages1 = [sum(x) / len(self.array_of_Xjs_upper[0]) for x in self.array_of_Xjs_upper]
        # gaps1 = [b - a for a, b in zip(self.Coverage_Upper_Bounds, actual_coverages1)]  # should all be negative
        # gaps1 = [num if num >= 0 else 0 for num in gaps1]
        # print("guy")
        # indices = [i for i in range(len(gaps)) if gaps[i] > 0]
        # cost_array = np.array(self.array_of_Xjs_lower)[indices].sum(axis=0)
        # if len(indices) > 0:
        #     pop_mean = sum(np.array(self.Coverage_Lower_Bounds)[indices])
        # else:
        #     pop_mean = 0
        # if not isinstance(cost_array, np.ndarray) and cost_array == 0.0:
        #     cost_array = [0.0]*len(self.array_of_Xjs_lower[0])
        # actual_coverages1 = [sum(x) / len(self.array_of_Xjs_upper[0]) for x in self.array_of_Xjs_upper]
        # gaps1 = [b - a for a, b in zip(self.Coverage_Upper_Bounds, actual_coverages1)]  # should all be negative
        # gaps1 = [num if num >= 0 else 0 for num in gaps1]
        # indices1 = [i for i in range(len(gaps1)) if gaps1[i] > 0]
        # cost_array1 = np.array(self.array_of_Xjs_upper)[indices1].sum(axis=0)
        # if len(indices) > 0:
        #     pop_mean1 = sum(np.array(self.Coverage_Upper_Bounds)[indices1])
        # else:
        #     pop_mean1 = 0
        # if not isinstance(cost_array1, np.ndarray) and cost_array1 == 0.0:
        #     cost_array1 = [0.0]*len(self.array_of_Xjs_lower[0])
        # final_cost_array = cost_array1 + cost_array
        #
        #
        # t_statistic, p_value = stats.ttest_1samp(a=final_cost_array,
        #                                          popmean=pop_mean + pop_mean1,
        #                                          alternative='less')
        # print("guy")
        # if len(indices) == 0:
        #     t_statistic, p_value = 1.0, 1.0
        # else:
        #     cost_array = np.array(self.array_of_Xjs_lower)[indices].sum(axis=0)
        #     t_statistic, p_value = stats.ttest_1samp(a=cost_array,
        #                                              popmean=sum(np.array(self.Coverage_Lower_Bounds)[indices]), alternative='less')
        #
        #
        # actual_coverages = [sum(x) / len(self.array_of_Xjs_upper[0]) for x in self.array_of_Xjs_upper]
        # gaps = [b - a for a, b in zip(self.Coverage_Upper_Bounds, actual_coverages)]  # should all be negative
        # gaps = [num if num >= 0 else 0 for num in gaps]
        # indices = [i for i in range(len(gaps)) if gaps[i] > 0]
        # if len(indices) == 0:
        #     t_statistic, p_value = 1.0, 1.0
        # else:
        #     cost_array = np.array(self.array_of_Xjs_upper)[indices].sum(axis=0) / len(self.array_of_Xjs_upper[0])
        #     t_statistic, p_value = stats.ttest_1samp(a=cost_array,
        #                                              popmean=sum(np.array(self.Coverage_Upper_Bounds)[indices]) / len(
        #                                                  self.array_of_Xjs_upper[0]), alternative='less')

        # if bound == 'L':
        #     new_list_for_hyp = [sum(x) for x in zip(*self.array_of_Xjs_lower)]
        #     mean = sum(self.Coverage_Lower_Bounds)
        #     t_statistic, p_value = stats.ttest_1samp(a=new_list_for_hyp, popmean=mean, alternative=tail)
        # else:
        #     mean = sum(self.Coverage_Upper_Bounds)
        #     new_list_for_hyp = [sum(x) for x in zip(*self.array_of_Xjs_upper)]
        #     t_statistic, p_value = stats.ttest_1samp(a=new_list_for_hyp,
        #                                              popmean=mean, alternative=tail)
        # return t_statistic, p_value

        # if bound == 'L':
        #     actual_coverages = [sum(x)/len(self.array_of_Xjs_lower[0]) for x in self.array_of_Xjs_lower]
        #     # index = self.find_largest_gap_index(actual_coverages, self.Coverage_Lower_Bounds)
        #     p_values = []
        #     for index in range(len(actual_coverages)):
        #     # new_list_for_hyp = [sum(x) for x in zip(*self.array_of_Xjs_lower)]
        #     # mean = sum(self.Coverage_Lower_Bounds)
        #         t_statistic, p_value = stats.ttest_1samp(a=self.array_of_Xjs_lower[index], popmean=self.Coverage_Lower_Bounds[index], alternative=tail)
        #         p_values.append(p_value)
        # else:
        #     actual_coverages = [sum(x)/len(self.array_of_Xjs_upper[0]) for x in self.array_of_Xjs_upper]
        #     # index = self.find_largest_gap_index(self.Coverage_Upper_Bounds, actual_coverages)
        #     p_values = []
        #     for index in range(len(actual_coverages)):
        #         t_statistic, p_value = stats.ttest_1samp(a=self.array_of_Xjs_upper[index],
        #                                                  popmean=self.Coverage_Upper_Bounds[index], alternative=tail)
        #         p_values.append(p_value)
        # p_value = sum(p_values) / len(p_values)
        # return t_statistic, p_value

    def _get_new_x_j(self, bound):
        """
        Calculating the new random variables for hypothesis test

        Args:
            bound: the bound we are calculating for

        Saves: an array of arrays, the inner array is an array of 0/1's and its average should
        as the coverage bound

        """
        self.array_of_Xjs_lower = []
        self.array_of_Xjs_upper = []
        num_iterations = len(self.Thresholds_For_Lower_Bounds)
        if bound == "L":
            with tqdm.tqdm(desc="Running Hypothesis Test for lower bound", total=num_iterations,
                           file=sys.stdout) as pbar:
                timer_start = timer()
                for theta_i in self.Thresholds_For_Lower_Bounds:
                    new_X_js = [0] * self.S_m_tot_size
                    for j, k_j in enumerate(self.S_m_us):
                        if k_j >= theta_i:
                            new_X_js[j] = 1
                    pbar.set_description(
                        f'Running Hypothesis Test for lower bound, Elapsed time: {timer() - timer_start:.3f} sec')
                    pbar.update()
                    self.array_of_Xjs_lower.append(new_X_js)
        else:
            with tqdm.tqdm(desc="Running Hypothesis Test for upper bound", total=num_iterations,
                           file=sys.stdout) as pbar:
                timer_start = timer()
                for theta_i in self.Thresholds_For_Upper_Bounds:
                    new_X_js = [0] * self.S_m_tot_size
                    for j, k_j in enumerate(self.S_m_us):
                        if k_j >= theta_i:
                            new_X_js[j] = 1
                    pbar.set_description(
                        f'Running Hypothesis Test for upper bound, Elapsed time: {timer() - timer_start:.3f} sec')
                    pbar.update()
                    self.array_of_Xjs_upper.append(new_X_js)


if __name__ == '__main__':
    import random


    def generate_random_numbers_between_0_and_1(n):
        """
        Generate n random numbers between 0 and 1.

        Args:
            n (int): The number of random numbers to generate.

        Returns:
            A list of n random numbers between 0 and 1.
        """
        return [random.random() for _ in range(n)]


    us_in_dist = sorted(generate_random_numbers_between_0_and_1(1000))
    us_window = sorted(generate_random_numbers_between_0_and_1(1000))
    C_num = 10
    delta = 0.0001

    detector = Shift_Detector(C_num, delta)

    detector.fit_lower_bound(us_in_dist)
    under_confidence_score = detector.detect_lower_bound_deviation(us_window)
    detector.visualize_lower_bound()

    detector.fit_upper_bound(us_in_dist)
    over_confidence_score = detector.detect_upper_bound_deviation(us_window)
    detector.visualize_upper_bound()
    exit()
