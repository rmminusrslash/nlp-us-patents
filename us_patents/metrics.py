from scipy import stats


class PearsonCorrelationCoefficient:
    """
    Computes the Pearson correlation coefficient. It measures the linear
    relationship between two datasets.

    Can be directly passed to HuggingFace Trainer's
    `compute_metrics` argument to monitor evaluation metrics
    while training.
    """

    name = "pearson_corr"

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.reshape(-1)
        score = stats.pearsonr(labels, predictions).statistic
        return {PearsonCorrelationCoefficient.name: score}
