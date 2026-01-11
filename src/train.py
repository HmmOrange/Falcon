from __future__ import annotations
import os, json, gc, yaml
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, log_loss
)

import lightgbm as lgb

# ============ LOAD CONFIG ============
# The script is run from the root directory, so the path is relative to the root
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

INPUT_DIR = CONFIG["INPUT_DIR"]
OUTPUT_DIR = CONFIG["OUTPUT_DIR"]
TRAINING_CONFIG = CONFIG["TRAINING"]
LGB_PARAMS_BASE = CONFIG["LGB_PARAMS"]
LGB_DEVICE_GPU = CONFIG["LGB_DEVICE"]["GPU"]
LGB_DEVICE_CPU = CONFIG["LGB_DEVICE"]["CPU"]
MLFLOW_CONFIG = CONFIG["MLFLOW"]

# Extract training params
N_SPLITS = TRAINING_CONFIG["N_SPLITS"]
EARLY_STOP = TRAINING_CONFIG["EARLY_STOP"]
SEED = TRAINING_CONFIG["SEED"]
THRESHOLD = TRAINING_CONFIG["THRESHOLD"]

# ============ UTILS ============
def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float = THRESHOLD) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    has_diff = (y_true.min() != y_true.max())
    
    return {
        "AUROC": roc_auc_score(y_true, y_prob) if has_diff else 0.5,
        "AUPRC": average_precision_score(y_true, y_prob) if has_diff else 0.0,
        "ACC": accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred, zero_division=0),
        "REC": recall_score(y_true, y_pred, zero_division=0),
        "F1":  f1_score(y_true, y_pred, zero_division=0),
        "LOGLOSS": log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)) if has_diff else 0.0,
    }

def select_features(X: np.ndarray, feat_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    indices_to_keep = []
    new_feat_names = []
    
    for i, name in enumerate(feat_names):
        if name.startswith("h_"):
            indices_to_keep.append(i)
            new_feat_names.append(name)
        
        elif name == "target_mean_time":
            indices_to_keep.append(i)
            new_feat_names.append(name)
            
    if not indices_to_keep:
        raise ValueError("Không tìm thấy features phù hợp (h_ hoặc target_mean_time).")
        
    X_filtered = X[:, indices_to_keep]
    
    return X_filtered, new_feat_names

def load_subset(prefix: str, feat_names_full: List[str]):
    path_X = os.path.join(INPUT_DIR, f"X_{prefix}.npy")
    path_y = os.path.join(INPUT_DIR, f"y_{prefix}.npy")
    path_g = os.path.join(INPUT_DIR, f"groups_{prefix}.npy")
    
    if not os.path.exists(path_X):
        raise FileNotFoundError(f"Missing file: {path_X}")
        
    X = np.load(path_X).astype(np.float32)
    y = np.load(path_y).astype(np.float32)
    g = np.load(path_g).astype(np.int64)
    
    X, feat_names_new = select_features(X, feat_names_full)
    
    return X, y, g, feat_names_new

# ============ MAIN ============    
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- MLFLOW SETUP ---
    mlflow.set_tracking_uri(MLFLOW_CONFIG["TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_CONFIG["EXPERIMENT_NAME"])

    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(TRAINING_CONFIG)
        mlflow.log_params(LGB_PARAMS_BASE)
        mlflow.log_artifact("config.yaml")

        # 1. Load Feature Names
        with open(os.path.join(INPUT_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
            feat_names_full = json.load(f)

        # 2. Load data
        X_train, y_train, groups_train, feat_names = load_subset("train", feat_names_full)
        X_holdout, y_holdout, _, _ = load_subset("holdout", feat_names_full)
        
        # ============================================================    
        # === TRAINING LOOP (K-FOLD) ===
        # ============================================================    
        gkf = GroupKFold(n_splits=N_SPLITS)
        
        oof_prob_train = np.zeros_like(y_train, dtype=np.float32)
        holdout_pred_sum = np.zeros(len(y_holdout), dtype=np.float32)
        
        fold_metrics: List[Dict[str, float]] = []
        feature_importance_df = pd.DataFrame(0, index=feat_names, columns=[f"fold_{i}" for i in range(1, N_SPLITS+1)], dtype=float)

        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups_train), 1):
            # Using nested runs for each fold for better organization
            with mlflow.start_run(nested=True, experiment_id=run.info.experiment_id) as fold_run:
                print(f"\n========== Fold {fold_id}/{N_SPLITS} (Run ID: {fold_run.info.run_id}) ==========")
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]

                # Calculate scale_pos_weight
                pos = float((y_tr == 1).sum())
                neg = float((y_tr == 0).sum())
                spw = max(1.0, (neg / (pos + 1e-9)))

                params = dict(LGB_PARAMS_BASE)
                params["scale_pos_weight"] = spw
                params["random_state"] = SEED

                # --- Train Model ---
                try:
                    model = lgb.LGBMClassifier(**params, **LGB_DEVICE_GPU)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
                    )
                except Exception as e:
                    print(f"[Fold {fold_id}] GPU training failed, fallback to CPU: {e}")
                    model = lgb.LGBMClassifier(**params, **LGB_DEVICE_CPU)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
                    )

                # --- Predictions ---
                prob_va = model.predict_proba(X_va, raw_score=False)[:, 1].astype(np.float32)
                oof_prob_train[va_idx] = prob_va
                
                holdout_pred_fold = model.predict_proba(X_holdout, raw_score=False)[:, 1].astype(np.float32)
                holdout_pred_sum += holdout_pred_fold

                # --- Evaluate and Log Metrics ---
                vm = evaluate_probs(y_va, prob_va, thr=THRESHOLD)
                fold_metrics.append(vm)
                mlflow.log_metrics({f"cv_{k}": v for k, v in vm.items()})
                print(f"[Fold {fold_id}] CV AUROC={{vm['AUROC']:.4f}}  F1={{vm['F1']:.4f}}")

                # --- Log Model and Importance ---
                booster = model.booster_
                try:
                    imp_gain = booster.feature_importance(importance_type="gain")
                except:
                    imp_gain = booster.feature_importance(importance_type="split")
                
                if len(imp_gain) == len(feat_names):
                     feature_importance_df[f"fold_{fold_id}"] = imp_gain
                
                mlflow.lightgbm.log_model(model, f"model_fold_{fold_id}")
                
                # Clear memory
                del X_tr, y_tr, X_va, y_va, model
                gc.collect()

        # ============================================================    
        # === AGGREGATE & LOG FINAL RESULTS ===
        # ============================================================    
        
        # 1. CV Mean Metrics
        print("\n========== 5-FOLD CV SUMMARY (Balanced Train Set) ==========")
        keys = ["AUROC", "AUPRC", "LOGLOSS", "ACC", "PRE", "REC", "F1"]
        avg_cv = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}
        mlflow.log_metrics({f"cv_mean_{k}": v for k, v in avg_cv.items()})
        for k in keys:
            print(f"Mean {k} = {avg_cv[k]:.4f}")
        
        # 2. Holdout Metrics
        print("\n========== HOLDOUT SET EVALUATION (Extreme Imbalance) ==========")
        final_holdout_prob = holdout_pred_sum / N_SPLITS
        holdout_metrics = evaluate_probs(y_holdout, final_holdout_prob, thr=THRESHOLD)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_metrics.items()})
        for k in keys:
            print(f"Holdout {k}: {holdout_metrics[k]:.4f}")
        
        # --- Save and Log Artifacts ---
        # OOF and Holdout predictions
        np.save(os.path.join(OUTPUT_DIR, "oof_prob_train.npy"), oof_prob_train)
        np.save(os.path.join(OUTPUT_DIR, "holdout_prob.npy"), final_holdout_prob)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "oof_prob_train.npy"))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "holdout_prob.npy"))
        
        # Metrics JSON
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "cv_folds": fold_metrics, 
                "cv_mean": avg_cv, 
                "holdout_metrics": holdout_metrics
            }, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(metrics_path)

        # Feature Importance
        feature_importance_df["avg_gain"] = feature_importance_df.mean(axis=1)
        feature_importance_df.sort_values("avg_gain", ascending=False, inplace=True)
        imp_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
        feature_importance_df.to_csv(imp_path)
        mlflow.log_artifact(imp_path)
        
        print(f"\nSaved all results to {OUTPUT_DIR}")
        print(f"MLflow experiment tracked successfully.")

if __name__ == "__main__":
    main()