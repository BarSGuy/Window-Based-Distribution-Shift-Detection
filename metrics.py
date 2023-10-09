import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix, precision_score


# https://github.com/tayden/ood-metrics

# ================================ metrics ================================ #

def precision(hard_preds, labels):
    """
    Calculates the precision score given binary predictions and ground truth labels.

    Args:
        preds (numpy.ndarray): Binary predictions.
        labels (numpy.ndarray): Ground truth labels.

    Returns:
        float: Precision score.
    """
    return precision_score(labels, hard_preds)


def recall(hard_preds, labels):
    """
    Calculates the recall score given binary predictions and ground truth labels.

    Args:
        hard_preds (numpy.ndarray): Binary predictions.
        labels (numpy.ndarray): Ground truth labels.

    Returns:
        float: Recall score.
    """
    return recall_score(labels, hard_preds)


def specificity(hard_preds, labels):
    """
    Calculates the specificity score given binary predictions and ground truth labels.

    Args:
        hard_preds (numpy.ndarray): Binary predictions.
        labels (numpy.ndarray): Ground truth labels.

    Returns:
        float: Specificity score.
    """
    tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
    return tn / (tn + fp)


def f1(hard_preds, labels):
    """
    Calculates the F1 score given binary predictions and ground truth labels.

    Args:
        hard_preds (numpy.ndarray): Binary predictions.
        labels (numpy.ndarray): Ground truth labels.

    Returns:
        float: F1 score.
    """
    return f1_score(hard_preds, labels)


def accuracy(hard_preds, labels):
    """
    Calculates the accuracy score given binary predictions and ground truth labels.

    Args:
        hard_preds (numpy.ndarray): Binary predictions.
        labels (numpy.ndarray): Ground truth labels.

    Returns:
        float: Accuracy score.
    """
    return accuracy_score(hard_preds, labels)


def auroc(preds, labels, pos_label=1):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)
    return auc(fpr, tpr)


def aupr(preds, labels, pos_label=1):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)


def fpr_at_95_tpr(preds, labels, pos_label=1):
    """Return the FPR when TPR is at minimum 95%.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def detection_error(preds, labels, pos_label=1):
    """Return the misclassification probability when TPR is 95%.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    # Get ratios of positives to negatives
    pos_ratio = sum(np.array(labels) == pos_label) / len(labels)
    neg_ratio = 1 - pos_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    # Calc is the (# of negatives * FNR) + (# of positives * FPR)
    _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))


# ===== function that calculates all the metrics ===== #
def calc_metricss(predictions, labels, pos_label=1, threshold=0.95):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.

    These metrics conform to how results are reported in the paper 'Enhancing The
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.

        preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

        labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
        pos_label: label of the positive class (1 by default)
    """

    def apply_threshold(array, threshold=threshold):
        """
        Given an array or list and a threshold, set values in the array or list to 1 if they are greater or equal to the
        threshold, and 0 otherwise.

        Args:
        - array (numpy array or list): The input array or list
        - threshold (float): The threshold value (default: 0.95)

        Returns:
        - A numpy array or list with the same shape as the input array or list, with values set to 1 or 0 based on the threshold
        """
        if isinstance(array, np.ndarray):
            return np.where(array >= threshold, 1, 0)
        else:
            return [1 if x >= threshold else 0 for x in array]

    hard_predictions = apply_threshold(predictions)
    #
    return {
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels, pos_label=pos_label),
        'detection_error': detection_error(predictions, labels, pos_label=pos_label),
        'auroc': auroc(predictions, labels, pos_label=pos_label),
        'aupr_in': aupr(predictions, labels, pos_label=pos_label),
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels], pos_label=pos_label),
        'precision': precision(hard_predictions, labels),
        'recall': recall(hard_predictions, labels),
        'specificity': specificity(hard_predictions, labels),
        'f1': f1(hard_predictions, labels),
        'accuracy': accuracy(hard_predictions, labels),
    }


def calc_metrics(predictions, labels, pos_label=1, threshold=0.95):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.

    These metrics conform to how results are reported in the paper 'Enhancing The
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.

        preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

        labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
        pos_label: label of the positive class (1 by default)
    """

    def apply_threshold(array, threshold=threshold):
        """
        Given an array or list and a threshold, set values in the array or list to 1 if they are greater or equal to the
        threshold, and 0 otherwise.

        Args:
        - array (numpy array or list): The input array or list
        - threshold (float): The threshold value (default: 0.95)

        Returns:
        - A numpy array or list with the same shape as the input array or list, with values set to 1 or 0 based on the threshold
        """
        if isinstance(array, np.ndarray):
            return np.where(array >= threshold, 1, 0)
        else:
            return [1 if x >= threshold else 0 for x in array]

    hard_predictions = apply_threshold(predictions)
    return {
        'fpr_at_95_tpr_out': fpr_at_95_tpr(predictions, labels, pos_label=1),
        'fpr_at_95_tpr_in': fpr_at_95_tpr([1 - a for a in predictions], [1 - a for a in labels], pos_label=1),
        'detection_error_out': detection_error(predictions, labels, pos_label=1),
        'detection_error_in': detection_error([1 - a for a in predictions], [1 - a for a in labels], pos_label=1),
        'auroc_out': auroc(predictions, labels, pos_label=1),
        # 'auroc_in': auroc([1 - a for a in predictions], [1 - a for a in labels], pos_label=1),
        'aupr_out': aupr(predictions, labels, pos_label=1),
        'aupr_in': aupr([1 - a for a in predictions], [1 - a for a in labels], pos_label=1),
        # 'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels], pos_label=pos_label),
        'precision_out': precision(hard_predictions, labels),
        'precision_in': precision([1 - a for a in hard_predictions], [1 - a for a in labels]),
        'recall_out': recall(hard_predictions, labels),
        # 'recall_in': recall([1 - a for a in hard_predictions], [1 - a for a in labels]),
        'specificity_out': specificity(hard_predictions, labels),
        # 'specificity_in': specificity([1 - a for a in hard_predictions], [1 - a for a in labels]),
        'f1_out': f1(hard_predictions, labels),
        'f1_in': f1([1 - a for a in hard_predictions], [1 - a for a in labels]),
        'accuracy': accuracy(hard_predictions, labels),
    }


# ========================================================================= #

# ================================ visualizations ================================ #
def plot_roc(preds, labels, title="Receiver operating characteristic"):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_pr(preds, labels, title="Precision recall curve"):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_barcode(preds, labels):
    """Plot a visualization showing inliers and outliers sorted by their prediction of novelty."""
    # the bar
    x = sorted([a for a in zip(preds, labels)], key=lambda x: x[0])
    x = np.array([[49, 163, 84] if a[1] == 1 else [173, 221, 142] for a in x])
    # x = np.array([a[1] for a in x]) # for bw image

    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary_r, interpolation='nearest')

    fig = plt.figure()

    # a horizontal barcode
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x.reshape((1, -1, 3)), **barprops)

    plt.show()


# ================================================================================ #
if __name__ == '__main__':
    def test_auroc():
        assert auroc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
        assert auroc([0.4, 0.3, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
        assert auroc([0.4, 0.3, 0.2, 0.1], [0, 1, 1, 0]) == 0.5
        assert auroc([0.4, 0.3, 0.2, 0.1], [-1, 1, 1, -1]) == 0.5
        assert auroc([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0]) == 0.0
        assert auroc([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1]) == 2. / 3


    test_auroc()


    def test_aupr():
        assert aupr([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
        assert round(aupr(list(range(10000)), [i % 2 for i in range(10000)]), 2) == 0.5


    test_aupr()


    def test_fpr_at_95_tpr():
        assert fpr_at_95_tpr([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 0.0
        assert fpr_at_95_tpr([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0]) == 1.0
        assert round(fpr_at_95_tpr(list(range(10000)), [i % 2 for i in range(10000)]), 2) == 0.95


    test_fpr_at_95_tpr()


    def test_detection_error():
        assert detection_error([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 0.0
        assert round(detection_error(list(range(100)), [1] * 3 + [0] * 97), 2) == 0.03
        assert round(detection_error(list(range(100)), [1] * 4 + [0] * 96), 2) == 0.04
        assert round(detection_error(list(range(10000)), [i % 2 for i in range(10000)]), 2) == 0.5


    test_detection_error()
