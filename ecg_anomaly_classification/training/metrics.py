from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_predictions(logits):
    return (logits > 0).long()


def get_metric(metric_name):
    if metric_name == "accuracy":
        return lambda y, logits: accuracy_score(y, get_predictions(logits))
    elif metric_name == "precision":
        return lambda y, logits: precision_score(y, get_predictions(logits))
    elif metric_name == "recall":
        return lambda y, logits: recall_score(y, get_predictions(logits))
    elif metric_name == "f1":
        return lambda y, logits: f1_score(y, get_predictions(logits))

    raise TypeError(f"Unknown metric name: {metric_name}")
