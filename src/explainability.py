import os
import shap
import matplotlib.pyplot as plt


def save_shap_summary(model, X, output_path="images/shap_summary.png"):
    """Save SHAP summary plot for model explainability."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return shap_values


def save_shap_bar(model, X, output_path="images/shap_bar.png"):
    """Save SHAP bar chart for top feature importance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=10)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return shap_values
