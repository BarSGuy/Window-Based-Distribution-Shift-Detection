from easydict import EasyDict as edict
from utils import *
from abc import ABC, abstractmethod
from Detector import Shift_Detector as SH
from Detector import *
import time
from alibi_detect.cd import MMDDrift, KSDrift
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib

# ====================================================================================== #
#                                   All methods                                          #
# ====================================================================================== #

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


class MyDetector(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # handle any additional arguments as needed
        self.c_num = kwargs.get('c_num')
        self.delta = kwargs.get('delta')
        self.temp = kwargs.get('temprature')
        self.uncertainty_mechanism = kwargs.get('uncertainty_mechanism')
        self.detector = SH(self.c_num, self.delta)

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

        self.detector.fit_lower_bound(X)

        self.desired_coverages = self.detector.get_desired_coverages()
        self.coverage_lower_bounds = self.detector.get_lower_bounds()
        sys.stdout = stdout

    def detect(self, X):
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        # apply the detector to X and return the results
        X = self.kappa_extractor(X, self.temp)
        start_time = time.time()
        self.p_value = self.detector.detect(X)
        end_time = time.time()
        self.actual_coverages_lower = self.detector.get_actual_coverage_for_lower()
        self.detection_time = end_time - start_time
        self.detection_parameters = edict({
            'desired_coverages': self.desired_coverages,
            'coverage_lower_bounds': self.coverage_lower_bounds,
            'detection_time': self.detection_time,
            'actual_coverages_lower': self.actual_coverages_lower,
            'final_score': 1 - self.p_value,
            'final_p_value': self.p_value
        })

        sys.stdout = stdout
        return self.detection_parameters.copy()


def plot_lower_bounds(cfg, should_hold, window_size, lower_bounds, actual_coverage_for_lower,
                 desired_coverages, suffix='', shift_extra_params=''):
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
    actual_coverage_for_lower_mean = np.mean(actual_coverage_for_lower, axis=0)
    actual_coverage_for_lower_std = np.std(actual_coverage_for_lower, axis=0) / np.sqrt(
        len(actual_coverage_for_lower))

    lower_bounds = np.mean(lower_bounds, axis=0)

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
    # violated_upper_indices = actual_coverage_for_upper_mean > upper_bounds

    # Plot the violations of the lower bound
    if any(violated_lower_indices):
        ax.scatter(np.array(x)[violated_lower_indices],
                   np.array(actual_coverage_for_lower_mean)[violated_lower_indices],
                   color='red', marker='o', s=100, label='Violations of Lower Bound')

    # add title and legend
    plt.title('Bounds and Coverage', fontsize=18)
    plt.legend(fontsize=10)

    # set x and y axis labels and limits
    plt.xlabel('Desired Coverages', fontsize=14)
    plt.ylabel('Coverage', fontsize=14)
    EPS = 0.01
    ymin = np.min([np.min(lower_bounds), np.min(actual_coverage_for_lower),
                   np.min(desired_coverages)])
    ymax = np.max([np.max(lower_bounds), np.max(actual_coverage_for_lower),
                   np.max(desired_coverages)])
    # ymax = np.max([np.max(upper_bounds), np.max(lower_bounds), np.max(actual_coverage_for_lower),
    #                np.max(actual_coverage_for_upper), np.max(desired_coverages)])
    plt.ylim(ymin - EPS, ymax + EPS)
    # add grid
    plt.grid(True, linestyle='--')
    # set x-axis tick locations and labels
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Save the plot as a PNG file
    filename = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}'
    if not os.path.exists(filename):
        os.makedirs(filename)
    if should_hold:
        plt.savefig(
            filename + f'/val_window_{window_size}' + suffix + '.png',
            dpi=300, bbox_inches='tight')
    else:
        plt.savefig(
            filename + f'/window_{window_size}' + suffix + '.png',
            dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


class Ks(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X):
        # train the detector on X and y
        self.detector = KSDrift(X, p_val=0.05)

    def detect(self, X):
        # apply the detector to X and return the results
        start_time = time.time()
        self.predictor = self.detector.predict(X, drift_type='batch')
        self.score = 1 - min(self.predictor['data']['p_val'])
        self.p_value = min(self.predictor['data']['p_val'])
        end_time = time.time()
        self.detection_time = end_time - start_time
        self.detection_parameters = edict({
            'detection_time': self.detection_time,
            'score': self.score,
            'p_value': self.p_value,
        })
        return self.detection_parameters.copy()


class Mmd(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = kwargs.get('device')

    def fit(self, X):
        # train the detector on X and y
        self.detector = MMDDrift(X, backend='pytorch', p_val=0.05, device=self.device)

    def detect(self, X):
        # apply the detector to X and return the results
        start_time = time.time()
        self.predictor = self.detector.predict(X)
        self.score = 1 - self.predictor['data']['p_val']
        self.p_value = self.predictor['data']['p_val']
        end_time = time.time()
        self.detection_time = end_time - start_time
        self.detection_parameters = edict({
            'detection_time': self.detection_time,
            'score': self.score,
            'p_value': self.p_value,
        })
        return self.detection_parameters.copy()


class Single_SR(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X):
        # X should be logits !!!
        # train the detector on X and y
        self.softmaxes_responses = get_softmax_responses(X)
        self.mean = sum(self.softmaxes_responses) / len(self.softmaxes_responses)
    def detect(self, X):
        start_time = time.time()
        self.softmaxes_responses_hat = get_softmax_responses(X)
        # self.p_value = self.two_sample_t_test(self.softmaxes_responses, self.softmaxes_responses_hat)
        self.p_value = self.one_sample_t_test(data=self.softmaxes_responses_hat, mean=self.mean)
        self.score = 1 - self.p_value
        end_time = time.time()
        self.detection_time = end_time - start_time
        self.detection_parameters = edict({
            'detection_time': self.detection_time,
            'score': self.score,
            'p_value': self.p_value,
        })
        return self.detection_parameters.copy()

    # def get_softmax_responses(self, logits_list, temperature=1.0):
    #     """
    #     Args:
    #         logits_list (list): List of PyTorch tensors or NumPy arrays, each containing a vector of logits.
    #         temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
    #
    #     Returns:
    #         List of PyTorch tensors or NumPy arrays, each containing a vector of softmax responses corresponding to the input logits.
    #     """
    #     if isinstance(logits_list[0], np.ndarray):
    #         logits_list = np.stack(logits_list)
    #
    #     softmax = F.softmax(torch.tensor(logits_list) / temperature, dim=-1)
    #     SR_list = torch.max(softmax, dim=1)[0].tolist()
    #
    #     return SR_list

    def two_sample_t_test(self, data1, data2):
        t_statistic, p_value = ttest_ind(data1, data2)
        return p_value

    def one_sample_t_test(self, data, mean):
        t_statistic, p_value = ttest_1samp(a=data, popmean=mean)
        return p_value



class Single_Ent(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X):
        # X should be logits !!!
        # train the detector on X and y
        self.softmaxes_entropies = get_entropy_of_softmax(X)
        self.mean = sum(self.softmaxes_entropies) / len(self.softmaxes_entropies)
# scipy.stats.ttest_1samp
    def detect(self, X):
        start_time = time.time()
        self.softmaxes_entropies_hat = get_entropy_of_softmax(X)
        # self.p_value = self.two_sample_t_test(self.softmaxes_entropies, self.softmaxes_entropies_hat)
        self.p_value = self.one_sample_t_test(data=self.softmaxes_entropies_hat, mean=self.mean)
        self.score = 1 - self.p_value
        end_time = time.time()
        self.detection_time = end_time - start_time
        self.detection_parameters = edict({
            'detection_time': self.detection_time,
            'score': self.score,
            'p_value': self.p_value,
        })
        return self.detection_parameters.copy()

    # def get_softmax_entropies(self, logits_list, temperature=1.0):
    #     if isinstance(logits_list[0], np.ndarray):
    #         logits_list = np.stack(logits_list)
    #
    #     softmax = F.softmax(torch.tensor(logits_list) / temperature, dim=-1)
    #     log_softmax = torch.log(softmax)
    #     entropy = -torch.sum(softmax * log_softmax, dim=1)
    #     normalized_entropy = entropy / torch.log(torch.tensor([softmax.shape[1]], dtype=torch.float))
    #
    #     return normalized_entropy.tolist()

    def two_sample_t_test(self, data1, data2):
        t_statistic, p_value = ttest_ind(data1, data2)
        return p_value
    def one_sample_t_test(self, data, mean):
        t_statistic, p_value = ttest_1samp(a=data, popmean=mean)
        return p_value


