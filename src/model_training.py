import pandas as pd
import optuna
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import ensemble, linear_model, tree, model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer


def benchmark_models(X, y) -> pd.DataFrame:
    """Benchmark multiple classification models using cross-validation."""
    models = [
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),
        linear_model.LogisticRegressionCV(solver="liblinear"),
        tree.DecisionTreeClassifier(),
        XGBClassifier(eval_metric="logloss", verbosity=0),
        LGBMClassifier(n_jobs=-1),
    ]

    cv_split = model_selection.ShuffleSplit(
        n_splits=10,
        test_size=0.3,
        train_size=0.6,
        random_state=0,
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    rows = []
    for model in models:
        name = model.__class__.__name__
        cv_results = model_selection.cross_validate(
            model,
            X,
            y,
            cv=cv_split,
            scoring=scoring,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "fit_time": cv_results["fit_time"].mean(),
                "accuracy": cv_results["test_accuracy"].mean(),
                "precision": cv_results["test_precision"].mean(),
                "recall": cv_results["test_recall"].mean(),
                "f1_score": cv_results["test_f1_score"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(by="f1_score", ascending=False)


def tune_xgboost_with_optuna(X_train, y_train, X_valid, y_valid, n_trials: int = 100):
    """Tune XGBoost hyperparameters using Optuna, optimizing recall."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }

        booster = xgb.train(params, dtrain)
        preds = booster.predict(dvalid)
        pred_labels = (preds >= 0.5).astype(int)
        return recall_score(y_valid, pred_labels)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params, study.best_trial.value


def train_xgboost(X_train, y_train, params=None):
    """Train final XGBoost classifier."""
    params = params or {}
    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
