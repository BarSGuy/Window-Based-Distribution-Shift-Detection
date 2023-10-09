from easydict import EasyDict as edict
from utils import *
from abc import ABC, abstractmethod
from Detector import Shift_Detector as SH
from Detector import *
import time
from alibi_detect.cd import MMDDrift, KSDrift
from scipy.stats import ttest_ind


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
        self.p_value = self.detector.detect(X)
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

            'final_score': 1 - self.p_value,
            'final_p_value': self.p_value
        })

        # # ====
        # self.score_lower = self.detector.detect_lower_bound_deviation(X)
        # self.actual_coverages_lower = self.detector.get_actual_coverage_for_lower()
        # self.score_upper = self.detector.detect_upper_bound_deviation(X)
        # self.actual_coverages_upper = self.detector.get_actual_coverage_for_upper()
        # end_time = time.time()
        #
        # self.p_value_upper = self.detector.detect_upper_bound_deviation(X, return_p_value=True)
        # self.p_value_lower = self.detector.detect_lower_bound_deviation(X, return_p_value=True)
        # self.detection_time = end_time - start_time
        # gap_lower = [a - b for a, b in zip(self.coverage_lower_bounds, self.actual_coverages_lower)]
        # gap_lower = [num if num >= 0 else 0 for num in gap_lower]
        #
        # gap_upper = [a - b for a, b in zip(self.actual_coverages_upper, self.coverage_upper_bounds)]
        # gap_upper = [num if num >= 0 else 0 for num in gap_upper]
        # sum_gaps = [a + b for a, b in zip(gap_lower, gap_upper)]
        # self.detection_parameters = edict({
        #     'desired_coverages': self.desired_coverages,
        #     'coverage_lower_bounds': self.coverage_lower_bounds,
        #     'coverage_upper_bounds': self.coverage_upper_bounds,
        #
        #     'detection_time': self.detection_time,
        #
        #     'score_lower': self.score_lower,
        #     'p_value_lower': self.p_value_lower,
        #     'actual_coverages_lower': self.actual_coverages_lower,
        #
        #     'score_upper': self.score_upper,
        #     'p_value_upper': self.p_value_upper,
        #     'actual_coverages_upper': self.actual_coverages_upper,
        #
        #     'final_score': max(self.score_lower, self.score_upper),
        #     # 'final_score': sum(sum_gaps) / len(sum_gaps),
        #     'final_p_value': min(self.p_value_lower, self.p_value_upper)
        # })
        sys.stdout = stdout
        return self.detection_parameters.copy()


def plot_bounds1(cfg, should_hold, upper_bounds, window_size, lower_bounds, actual_coverage_for_lower,
                 actual_coverage_for_upper,
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
    # ax.plot(x, upper_bounds, color='blue', linestyle='dashed', label='Upper Bound', linewidth=3, alpha=0.8)
    # ax.plot(x, actual_coverage_for_upper_mean, color='blue', label='Actual Coverage for Upper')
    # ax.fill_between(x, actual_coverage_for_upper_mean - actual_coverage_for_upper_std,
    #                 actual_coverage_for_upper_mean + actual_coverage_for_upper_std,
    #                 color='blue', alpha=0.2)
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
    # if any(violated_upper_indices):
    #     ax.scatter(np.array(x)[violated_upper_indices],
    #                np.array(actual_coverage_for_upper_mean)[violated_upper_indices],
    #                color='blue', marker='o', s=100, label='Violations of Upper Bound')

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

    def detect(self, X):
        start_time = time.time()
        self.softmaxes_responses_hat = get_softmax_responses(X)
        self.p_value = self.two_sample_t_test(self.softmaxes_responses, self.softmaxes_responses_hat)
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


class Single_Ent(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X):
        # X should be logits !!!
        # train the detector on X and y
        self.softmaxes_entropies = get_entropy_of_softmax(X)

    def detect(self, X):
        start_time = time.time()
        self.softmaxes_entropies_hat = get_entropy_of_softmax(X)
        self.p_value = self.two_sample_t_test(self.softmaxes_entropies, self.softmaxes_entropies_hat)
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


def save_data_my_detector_imagenet(cfg, result_dicts, window_size, threshold=0.95, suffix='', shift_extra_params=''):
    plot_bounds1(cfg, should_hold=True, window_size=window_size,
                 upper_bounds=get_values_from_list_dict(result_dicts, ['in_vs_in', str(window_size), 'my_detector',
                                                                       'coverage_upper_bounds']),
                 lower_bounds=get_values_from_list_dict(result_dicts, ['in_vs_in', str(window_size), 'my_detector',
                                                                       'coverage_lower_bounds']),
                 actual_coverage_for_lower=get_values_from_list_dict(result_dicts,
                                                                     ['in_vs_in', str(window_size), 'my_detector',
                                                                      'actual_coverages_lower']),
                 actual_coverage_for_upper=get_values_from_list_dict(result_dicts,
                                                                     ['in_vs_in', str(window_size), 'my_detector',
                                                                      'actual_coverages_upper']),
                 desired_coverages=get_values_from_list_dict(result_dicts,
                                                             ['in_vs_in', str(window_size), 'my_detector',
                                                              'desired_coverages'])[0], suffix=suffix,
                 shift_extra_params=shift_extra_params)

    plot_bounds1(cfg, should_hold=False, window_size=window_size,
                 upper_bounds=get_values_from_list_dict(result_dicts, ['in_vs_out', str(window_size), 'my_detector',
                                                                       'coverage_upper_bounds']),
                 lower_bounds=get_values_from_list_dict(result_dicts, ['in_vs_out', str(window_size), 'my_detector',
                                                                       'coverage_lower_bounds']),
                 actual_coverage_for_lower=get_values_from_list_dict(result_dicts,
                                                                     ['in_vs_out', str(window_size), 'my_detector',
                                                                      'actual_coverages_lower']),
                 actual_coverage_for_upper=get_values_from_list_dict(result_dicts,
                                                                     ['in_vs_out', str(window_size), 'my_detector',
                                                                      'actual_coverages_upper']),
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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
    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
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

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + f'/imagenet' + f'/imagenet_vs_{cfg.IMAGENET.EXPERIMENT.OOD}' + shift_extra_params + f'_{cfg.IMAGENET.MODEL}/drift_single_entropies_window_{window_size}' + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


# =================================================================

def save_data_my_detector(cfg, result_dicts, window_size, threshold=0.95, suffix_for_file_name=''):
    plot_bounds(cfg, should_hold=True, window_size=window_size,
                upper_bounds=get_values_from_list_dict(result_dicts, ['in_vs_in', str(window_size), 'my_detector',
                                                                      'coverage_upper_bounds']),
                lower_bounds=get_values_from_list_dict(result_dicts, ['in_vs_in', str(window_size), 'my_detector',
                                                                      'coverage_lower_bounds']),
                actual_coverage_for_lower=get_values_from_list_dict(result_dicts,
                                                                    ['in_vs_in', str(window_size), 'my_detector',
                                                                     'actual_coverages_lower']),
                actual_coverage_for_upper=get_values_from_list_dict(result_dicts,
                                                                    ['in_vs_in', str(window_size), 'my_detector',
                                                                     'actual_coverages_upper']),
                desired_coverages=get_values_from_list_dict(result_dicts,
                                                            ['in_vs_in', str(window_size), 'my_detector',
                                                             'desired_coverages'])[0],
                suffix_for_file_name=suffix_for_file_name)

    plot_bounds(cfg, should_hold=False, window_size=window_size,
                upper_bounds=get_values_from_list_dict(result_dicts, ['in_vs_out', str(window_size), 'my_detector',
                                                                      'coverage_upper_bounds']),
                lower_bounds=get_values_from_list_dict(result_dicts, ['in_vs_out', str(window_size), 'my_detector',
                                                                      'coverage_lower_bounds']),
                actual_coverage_for_lower=get_values_from_list_dict(result_dicts,
                                                                    ['in_vs_out', str(window_size), 'my_detector',
                                                                     'actual_coverages_lower']),
                actual_coverage_for_upper=get_values_from_list_dict(result_dicts,
                                                                    ['in_vs_out', str(window_size), 'my_detector',
                                                                     'actual_coverages_upper']),
                desired_coverages=get_values_from_list_dict(result_dicts,
                                                            ['in_vs_out', str(window_size), 'my_detector',
                                                             'desired_coverages'])[0],
                suffix_for_file_name=suffix_for_file_name)

    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'my_detector', 'final_score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'my_detector', 'final_score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/my_detector_window_{window_size}' + suffix_for_file_name + f'.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/my_detector_window_{window_size}' + suffix_for_file_name + f'.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/my_detector_window_{window_size}' + suffix_for_file_name + f'.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/my_detector_window_{window_size}' + suffix_for_file_name + f'.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_ks_embs(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_ks_embs', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_ks_embs', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_ks_embs_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_ks_embs_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_ks_embs_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_ks_embs_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_ks_logits(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_ks_logits', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_ks_logits', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_ks_logits_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_ks_logits_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_ks_logits_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_ks_logits_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_ks_softmaxes(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_ks_softmaxes', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_ks_softmaxes', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_ks_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_ks_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_ks_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_ks_softmaxes_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_mmd_embs(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_mmd_embs', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_mmd_embs', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_mmd_embs_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_mmd_embs_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_mmd_embs_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_mmd_embs_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_mmd_logits(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_mmd_logits', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_mmd_logits', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_mmd_logits_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_mmd_logits_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_mmd_logits_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_mmd_logits_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_mmd_softmaxes(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_mmd_softmaxes', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_mmd_softmaxes', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_mmd_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_mmd_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_mmd_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_mmd_softmaxes_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_single_softmaxes(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_single_softmaxes', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_single_softmaxes', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_single_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_single_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_single_softmaxes_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_single_softmaxes_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


def save_data_drift_single_entropies(cfg, result_dicts, window_size, threshold=0.95):
    preds_in_vs_in = get_values_from_list_dict(result_dicts,
                                               ['in_vs_in', str(window_size), 'drift_single_entropies', 'score'])
    labels_in_vs_in = [0] * len(preds_in_vs_in)
    preds_in_vs_out = get_values_from_list_dict(result_dicts,
                                                ['in_vs_out', str(window_size), 'drift_single_entropies', 'score'])
    labels_in_vs_out = [1] * len(preds_in_vs_out)
    all_preds = np.array(concatenate_lists(preds_in_vs_in, preds_in_vs_out))
    all_labels = np.array(concatenate_lists(labels_in_vs_in, labels_in_vs_out))

    metrics_dict = get_bootstrapped_metrics_with_stds(num_bootstrap_runs=cfg.EXPERIMENT.NUM_BOOTSTRAP_RUNS,
                                                      all_preds=all_preds,
                                                      all_labels=all_labels, threshold=threshold)
    csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_{cfg.MODEL.NAME}/drift_single_entropies_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_STD_{cfg.GENERATING_OOD.GAUSS_NOISE.STD}_{cfg.MODEL.NAME}/drift_single_entropies_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Angle_{cfg.GENERATING_OOD.ROTATE.ANGLE}_{cfg.MODEL.NAME}/drift_single_entropies_window_{window_size}.csv'
    if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
        csv_path = cfg.PATH_TO_DICTIONARIES + f'/{cfg.IN_DISTRIBUTION}' + f'/{cfg.IN_DISTRIBUTION}_vs_{cfg.OUT_OF_DISTRIBUTION}_Factor_{cfg.GENERATING_OOD.ZOOM.FACTOR}_{cfg.MODEL.NAME}/drift_single_entropies_window_{window_size}.csv'
    save_dict_to_csv(metrics_dict, csv_path)


# =================================================================

# =========================== general loop =========================== #
def general_experiment_loop(cfg, outputs_in, outputs_out):
    # Initialize results_dict
    results_dict = edict({})

    # Shuffle input lists to ensure randomness
    outputs_in = shuffle_list(outputs_in)
    outputs_out = shuffle_list(outputs_out)

    # Split the in-distribution data into train and validation sets
    outputs_in_train, outputs_in_val = split_array(np.array(outputs_in), len(outputs_in) // 2)
    cfg.logging.info(f"Splitting training outputs to training and validation sets")
    # Fit detectors on the in-distribution training set


# ==================================================================== #


def Ours():
    pass


def ks_embs():
    pass


def ks_logits():
    pass


def ks_softmaxes():
    pass


def mmd_embs():
    pass


def mmd_logits():
    pass


def mmd_softmaxes():
    pass
