import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)


def print_classification_metrics(model, X_test, y_test):
    """Print classification report."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred


def save_confusion_matrix(model, X_test, y_test, output_path="images/confusion_matrix.png"):
    """Save confusion matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot()
    plt.title("Normalized Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_roc_pr_curves(model, X_test, y_test, output_path="images/roc_pr_curves.png"):
    """Save ROC and Precision-Recall curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    no_skill = sum(y_test == 1) / len(y_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([0, 1], [0, 1], linestyle="--", label="Baseline")
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot([0, 1], [no_skill, no_skill], linestyle="--", label="Baseline")
    axes[1].plot(recall, precision, label=f"AP = {avg_precision:.2f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
