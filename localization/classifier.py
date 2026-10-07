"""Cascaded five-class location classification and evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import parameter, validate_parameters

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
import time
import os
import joblib

warnings.filterwarnings("ignore")


class CascadeXGBClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, n_estimators, random_state):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.l1_model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        self.l2a_model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.l2b_model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    def _map_label_l1(self, y):
        y_new = y.copy()
        y_new[np.isin(y, [0, 4])] = 0
        y_new[np.isin(y, [1, 2])] = 1
        y_new[y == 3] = 2
        return y_new

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(y, "values") else y
        self.classes_ = np.unique(y_arr)
        y_l1 = self._map_label_l1(y_arr)
        self.l1_model.fit(X_arr, y_l1)
        mask_infra = np.isin(y_arr, [1, 2])
        if np.sum(mask_infra) > 0:
            X_infra = X_arr[mask_infra]
            y_infra = y_arr[mask_infra]
            y_infra_mapped = np.where(y_infra == 1, 0, 1)
            self.l2a_model.fit(X_infra, y_infra_mapped)
        mask_deep = np.isin(y_arr, [0, 4])
        if np.sum(mask_deep) > 0:
            X_deep = X_arr[mask_deep]
            y_deep = y_arr[mask_deep]
            y_deep_mapped = np.where(y_deep == 0, 0, 1)
            self.l2b_model.fit(X_deep, y_deep_mapped)
        return self

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else X
        p_l1 = self.l1_model.predict_proba(X_arr)
        p_l2a = (
            self.l2a_model.predict_proba(X_arr)
            if hasattr(self.l2a_model, "classes_")
            else np.zeros((len(X_arr), 2))
        )
        p_l2b = (
            self.l2b_model.predict_proba(X_arr)
            if hasattr(self.l2b_model, "classes_")
            else np.zeros((len(X_arr), 2))
        )
        p_final = np.zeros((len(X), 5))
        p_final[:, 0] = p_l1[:, 0] * p_l2b[:, 0]
        p_final[:, 1] = p_l1[:, 1] * p_l2a[:, 0]
        p_final[:, 2] = p_l1[:, 1] * p_l2a[:, 1]
        p_final[:, 3] = p_l1[:, 2]
        p_final[:, 4] = p_l1[:, 0] * p_l2b[:, 1]
        return p_final

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def main():
    print("--- Starting Cascaded XGBoost Evaluation Pipeline ---")
    import argparse

    parser = argparse.ArgumentParser(description="Cascaded location classification")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    validate_parameters(
        "localization.n_estimators", "localization.seed", "localization.cv_folds"
    )
    data_path = args.data
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(data_path)
    df_train = df[df["subset"] == "train"].reset_index(drop=True)
    df_test = df[df["subset"] == "test"].reset_index(drop=True)
    drop_cols = ["id", "subset", "label"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    X_train_raw = df_train[feature_cols]
    y_train = df_train["label"].astype(int)
    X_test_raw = df_test[feature_cols]
    y_test = df_test["label"].astype(int)
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed), columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=feature_cols)
    model = CascadeXGBClassifier(
        parameter("localization.n_estimators"), parameter("localization.seed")
    )
    print("\nRunning Cross-validation OOF evaluation...")
    skf = StratifiedKFold(
        n_splits=parameter("localization.cv_folds"),
        shuffle=True,
        random_state=parameter("localization.seed"),
    )
    y_prob_cv = cross_val_predict(
        model, X_train_scaled, y_train, cv=skf, method="predict_proba", n_jobs=1
    )
    y_pred_cv = np.argmax(y_prob_cv, axis=1)
    df_train_cv = df_train[["id", "label"]].copy()
    df_train_cv["predict"] = y_pred_cv
    for i, class_name in enumerate(
        [
            "Prob_Basal",
            "Prob_Brainstem",
            "Prob_Cerebellum",
            "Prob_Lobar",
            "Prob_Thalamus",
        ]
    ):
        df_train_cv[class_name] = y_prob_cv[:, i]
    oof_path = os.path.join(output_dir, "cv_oof_predictions.csv")
    df_train_cv.to_csv(oof_path, index=False)
    print(f"Saved OOF predictions to {oof_path}")
    b_acc_list = []
    pr_auc_list = []
    class_metrics_cv = {c: {"p": [], "r": [], "f1": []} for c in range(5)}
    for train_idx, test_idx in skf.split(X_train_scaled, y_train):
        X_tr, X_val = (X_train_scaled.iloc[train_idx], X_train_scaled.iloc[test_idx])
        y_tr, y_val = (y_train.iloc[train_idx], y_train.iloc[test_idx])
        m = CascadeXGBClassifier(
            parameter("localization.n_estimators"), parameter("localization.seed")
        )
        m.fit(X_tr, y_tr)
        p_val = m.predict_proba(X_val)
        y_p = np.argmax(p_val, axis=1)
        b_acc_list.append(balanced_accuracy_score(y_val, y_p))
        y_val_bin = label_binarize(y_val, classes=[0, 1, 2, 3, 4])
        pr_auc_list.append(average_precision_score(y_val_bin, p_val, average="macro"))
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, y_p, labels=[0, 1, 2, 3, 4], zero_division=0
        )
        for i in range(5):
            class_metrics_cv[i]["p"].append(p[i])
            class_metrics_cv[i]["r"].append(r[i])
            class_metrics_cv[i]["f1"].append(f1[i])
    cv_report_path = os.path.join(output_dir, "cv_report.txt")
    with open(cv_report_path, "w", encoding="utf-8") as f:
        f.write("=== CASCADED XGBOOST CROSS-VALIDATION REPORT ===\n\n")
        f.write(
            f"Balanced Accuracy: {np.mean(b_acc_list):.4f} +/- {np.std(b_acc_list):.4f}\n"
        )
        f.write(
            f"Macro PR AUC:      {np.mean(pr_auc_list):.4f} +/- {np.std(pr_auc_list):.4f}\n\n"
        )
        f.write("Class-Wise Breakdown:\n")
        class_names = ["Basal", "Brainstem", "Cerebellum", "Lobar", "Thalamus"]
        for i, cname in enumerate(class_names):
            f.write(f" [{cname}]\n")
            f.write(
                f"  Precision: {np.mean(class_metrics_cv[i]['p']):.4f} +/- {np.std(class_metrics_cv[i]['p']):.4f}\n"
            )
            f.write(
                f"  Recall:    {np.mean(class_metrics_cv[i]['r']):.4f} +/- {np.std(class_metrics_cv[i]['r']):.4f}\n"
            )
            f.write(
                f"  F1-Score:  {np.mean(class_metrics_cv[i]['f1']):.4f} +/- {np.std(class_metrics_cv[i]['f1']):.4f}\n\n"
            )
    print("\nTraining on Full Training Set for Final Evaluation...")
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, os.path.join(output_dir, "cascaded_xgboost_model.pkl"))
    joblib.dump(imputer, os.path.join(output_dir, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    y_prob_test = model.predict_proba(X_test_scaled)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    df_test_res = df_test[["id", "label"]].copy()
    df_test_res["predict"] = y_pred_test
    for i, class_name in enumerate(
        [
            "Prob_Basal",
            "Prob_Brainstem",
            "Prob_Cerebellum",
            "Prob_Lobar",
            "Prob_Thalamus",
        ]
    ):
        df_test_res[class_name] = y_prob_test[:, i]
    test_pred_path = os.path.join(output_dir, "test_predictions.csv")
    df_test_res.to_csv(test_pred_path, index=False)
    print(f"Saved Test predictions to {test_pred_path}")
    test_b_acc = balanced_accuracy_score(y_test, y_pred_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    test_pr_auc = average_precision_score(y_test_bin, y_prob_test, average="macro")
    test_acc = accuracy_score(y_test, y_pred_test)
    tp, tr, tf1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred_test)
    test_report_path = os.path.join(output_dir, "evaluation_report_test.txt")
    with open(test_report_path, "w", encoding="utf-8") as f:
        f.write("=== CASCADED XGBOOST INDEPENDENT TEST REPORT ===\n\n")
        f.write(f"Overall Accuracy:  {test_acc:.4f}\n")
        f.write(f"Balanced Accuracy: {test_b_acc:.4f}\n")
        f.write(f"Macro PR AUC:      {test_pr_auc:.4f}\n\n")
        f.write("Class-Wise Breakdown:\n")
        for i, cname in enumerate(class_names):
            f.write(f" [{cname}]\n")
            f.write(f"  Precision: {tp[i]:.4f}\n")
            f.write(f"  Recall:    {tr[i]:.4f}\n")
            f.write(f"  F1-Score:  {tf1[i]:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print("Cascaded XGBoost execution cleanly finished.")


if __name__ == "__main__":
    main()
