from __future__ import annotations
import os
import random
import json
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# ============================
# CONFIG
# ============================

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# General Config
OUTPUT_DIR = CONFIG["OUTPUT_DIR"]
SEED = CONFIG["TRAINING"]["SEED"]

# Prepare Data Config
PREP_CONFIG = CONFIG["PREPARE_DATA"]
INPUT_CSV_PATH = PREP_CONFIG["INPUT_CSV_PATH"]
MEAN_TIME_PATH = PREP_CONFIG["MEAN_TIME_PATH"]

# Columns
COL_ACCOUNT_ID = PREP_CONFIG["COLUMNS"]["ACCOUNT_ID"]
COL_LEVEL_ID = PREP_CONFIG["COLUMNS"]["LEVEL_ID"]
COL_STATUS = PREP_CONFIG["COLUMNS"]["STATUS"]
COL_DURATION = PREP_CONFIG["COLUMNS"]["DURATION"]

N_CONT_HIST_FEATURES = PREP_CONFIG["N_CONT_HIST_FEATURES"]

# Dataset params
SEQ_LEN = PREP_CONFIG["SEQ_LEN"]
SAMPLING_CONFIG = PREP_CONFIG["SAMPLING"]
NEG_HISTORY_BACKSHIFT_MIN = SAMPLING_CONFIG["NEG_HISTORY_BACKSHIFT_MIN"]
NEG_HISTORY_BACKSHIFT_MAX = SAMPLING_CONFIG["NEG_HISTORY_BACKSHIFT_MAX"]
TRAIN_NEG_PER_POS = SAMPLING_CONFIG["TRAIN_NEG_PER_POS"]
TRAIN_SPLIT_RATIO = PREP_CONFIG["TRAIN_SPLIT_RATIO"]

# System
N_WORKERS_CONFIG = PREP_CONFIG["N_WORKERS"]
if N_WORKERS_CONFIG == -1:
    N_WORKERS = (os.cpu_count() or 1)
else:
    N_WORKERS = N_WORKERS_CONFIG

# Seed
random.seed(SEED)
np.random.seed(SEED)

# Load mean_time
mean_time = joblib.load(MEAN_TIME_PATH)

# ======================
# 1) Loaders & Scalers
# =====================

def load_players_from_csv(csv_path: str) -> List[np.ndarray]:
    feature_cols_ordered = [COL_LEVEL_ID, COL_STATUS, COL_DURATION]
    all_cols_to_read = [COL_ACCOUNT_ID] + feature_cols_ordered
    
    try:
        df = pd.read_csv(csv_path, usecols=all_cols_to_read)
    except ValueError as e:
        print(f"--- ERROR ---")
        print(f"Could not find columns: {all_cols_to_read}")
        raise
        
    players_list = []
    grouped = df.groupby(COL_ACCOUNT_ID)
    
    for _, group_df in tqdm(grouped, desc="Grouping players", total=len(grouped)):
        player_array = group_df[feature_cols_ordered].to_numpy().astype(np.float32)
        players_list.append(player_array)
        
    return players_list


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)

def compute_hist_scaler(players: List[np.ndarray]) -> Tuple[Scaler, int]:
    concat = np.concatenate(players, axis=0) if len(players) else np.zeros((0,3))
    cont = concat[:, 1:3].astype(np.float32) if concat.size else np.zeros((0,N_CONT_HIST_FEATURES), np.float32)
    mean = cont.mean(axis=0) if cont.size else np.zeros(N_CONT_HIST_FEATURES, np.float32)
    std  = cont.std(axis=0)  if cont.size else np.ones(N_CONT_HIST_FEATURES, np.float32)
    scaler = Scaler(mean=mean.astype(np.float32), std=std.astype(np.float32))
    max_level = int(concat[:, 0].max()) if concat.size else 1
    return scaler, max_level

def compute_scaler_from_seq(seq: np.ndarray) -> Scaler:
    if seq.size == 0:
        return Scaler(mean=np.zeros(N_CONT_HIST_FEATURES, np.float32), std=np.ones(N_CONT_HIST_FEATURES, np.float32))
    cont = seq[:, 1:3].astype(np.float32)
    mean = cont.mean(axis=0).astype(np.float32)
    std  = cont.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return Scaler(mean=mean, std=std)


# ======================================
# 2) Dataset building
# ======================================
def sort_by_level_then_status(seq: np.ndarray) -> np.ndarray:
    if seq.size == 0: return seq
    idx0 = np.arange(seq.shape[0])
    order = np.lexsort((idx0, seq[:,1], seq[:,0]))
    return seq[order]

def make_hist_window(seq_before: np.ndarray, seq_len: int, scaler: Scaler) -> Tuple[np.ndarray, np.ndarray, int]:
    seq_before = sort_by_level_then_status(seq_before)
    if seq_before.size == 0:
        levels = np.zeros((1,), np.int64)
        cont   = np.zeros((1, N_CONT_HIST_FEATURES), np.float32) 
        length = 1
    else:
        levels = seq_before[:,0].astype(np.int64)
        cont   = seq_before[:,1:3].astype(np.float32) 
        cont   = scaler.transform(cont)
        length = len(levels)
    if length > seq_len:
        levels = levels[-seq_len:]
        cont   = cont[-seq_len:]
        length = seq_len
    pad = seq_len - length
    if pad > 0:
        levels = np.concatenate([levels, np.zeros(pad, dtype=np.int64)])
        cont   = np.concatenate([cont,   np.zeros((pad, N_CONT_HIST_FEATURES), dtype=np.float32)], axis=0)
    return levels, cont, length

# === multiprocessing globals ===
_G_SEQ_LEN = None
_G_HIST_SCALER: Optional[Scaler] = None
_G_NEG_PER_POS: Optional[int] = None
_G_SEED = 42
_G_NEG_BACK_MIN = 1
_G_NEG_BACK_MAX = 2
_G_MEAN_TIME = None 

def _init_worker(seq_len, hist_scaler: Scaler, 
                 neg_per_pos: Optional[int], 
                 base_seed: int,
                 neg_back_min: int, neg_back_max: int,
                 mean_time_arr): 
    global _G_SEQ_LEN, _G_HIST_SCALER, _G_NEG_PER_POS, _G_SEED, _G_NEG_BACK_MIN, _G_NEG_BACK_MAX, _G_MEAN_TIME
    _G_SEQ_LEN = seq_len
    _G_HIST_SCALER = hist_scaler
    _G_NEG_PER_POS = neg_per_pos
    _G_SEED = base_seed
    _G_NEG_BACK_MIN = neg_back_min
    _G_NEG_BACK_MAX = neg_back_max
    _G_MEAN_TIME = mean_time_arr 

def _process_one_player(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pid, seq = args
    rng = np.random.default_rng(_G_SEED + pid)

    ps = sort_by_level_then_status(seq) 
    if ps.size == 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 
                np.zeros((0,), dtype=np.int64))

    player_scaler = compute_scaler_from_seq(ps)
    lv = ps[:,0].astype(np.int64)
    st = ps[:,1].astype(np.int64)

    uniq, first_idx = np.unique(lv, return_index=True)
    passed = np.zeros(len(uniq), dtype=bool)
    ends = np.r_[first_idx[1:], len(lv)]
    for i, (s, e) in enumerate(zip(first_idx, ends)):
        if np.any(st[s:e] == 1):
            passed[i] = True

    last_idx = len(uniq) - 1
    last_level = int(uniq[last_idx])

    if len(uniq) > 1:
        neg_mask_idx = np.arange(len(uniq) - 1)
        neg_targets_all = uniq[neg_mask_idx[passed[:len(uniq)-1]]].tolist()
    else:
        neg_targets_all = []

    pos_targets = [last_level]

    if _G_NEG_PER_POS is None:
        neg_targets = neg_targets_all
    else:
        k = _G_NEG_PER_POS * max(1, len(pos_targets))
        if len(neg_targets_all) > k:
            neg_targets = rng.choice(neg_targets_all, size=k, replace=False).tolist()
        else:
            neg_targets = neg_targets_all

    hist_levels_L, hist_cont_L, hist_len_L = [], [], []
    tgt_ids_L, labels_L = [], []
    tgt_means_L = []

    N_unique = len(uniq)
    use_mid_rule = (N_unique / 2.0) > (_G_SEQ_LEN + 5)

    def add_sample(target_level: int, label: int):
        if label == 0:
            if use_mid_rule and _G_NEG_PER_POS is not None: 
                mid_idx = N_unique // 2
                start_idx = max(0, mid_idx - (_G_SEQ_LEN - 1))
                chosen_levels = set(int(x) for x in uniq[start_idx:mid_idx+1])
                mask_mid = np.isin(lv, list(chosen_levels)) & (lv != int(target_level))
                seq_before = ps[mask_mid]
            else:
                if _G_NEG_PER_POS is not None:
                     delta = int(np.random.randint(_G_NEG_BACK_MIN, _G_NEG_BACK_MAX + 1)) if _G_NEG_BACK_MAX > 0 else 0
                     cut_level = max(1, int(target_level) - delta + 1)
                else:
                     cut_level = int(target_level)
                
                mask = (lv < cut_level)
                seq_before = ps[mask]
        else:
            cut_level = int(target_level)
            mask = lv < cut_level
            seq_before = ps[mask]

        h_levels, h_cont, h_len = make_hist_window(seq_before, _G_SEQ_LEN, player_scaler)

        hist_levels_L.append(h_levels)
        hist_cont_L.append(h_cont)
        hist_len_L.append(h_len)
        tgt_ids_L.append(int(target_level))
        labels_L.append(int(label))
        
        idx = int(target_level) - 1
        val = 0.0
        if _G_MEAN_TIME is not None and 0 <= idx < len(_G_MEAN_TIME):
             val = float(_G_MEAN_TIME[idx])
        tgt_means_L.append(val)

    for tl in neg_targets:
        add_sample(int(tl), 0)
    for tl in pos_targets:
        add_sample(int(tl), 1)

    if len(labels_L) == 0:
        return (np.zeros((0,), np.int64), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64), np.zeros((0,), np.int64),
                np.zeros((0,), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))

    hist_levels = np.stack(hist_levels_L).astype(np.int64)
    hist_cont   = np.stack(hist_cont_L).astype(np.float32)
    hist_len    = np.array(hist_len_L, dtype=np.int64)
    tgt_ids     = np.array(tgt_ids_L, dtype=np.int64)
    tgt_means   = np.array(tgt_means_L, dtype=np.float32) 
    labels      = np.array(labels_L, dtype=np.float32)
    groups      = np.full(len(labels_L), pid, dtype=np.int64)
    
    return hist_levels, hist_cont, hist_len, tgt_ids, tgt_means, labels, groups

def build_dataset_fast(players: List[np.ndarray],
                       seq_len: int,
                       hist_scaler: Scaler,
                       neg_per_pos: Optional[int],
                       n_workers: int,
                       mean_time_arr) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    args_iter = list(enumerate(players))
    results = []
    
    desc = "Building dataset"
    if n_workers and n_workers > 0:
        with mp.get_context("fork" if os.name != "nt" else "spawn").Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(seq_len, hist_scaler,
                      neg_per_pos,
                      SEED,
                      NEG_HISTORY_BACKSHIFT_MIN, NEG_HISTORY_BACKSHIFT_MAX,
                      mean_time_arr)
        ) as pool:
            for out in tqdm(pool.imap_unordered(_process_one_player, args_iter), total=len(args_iter), desc=desc):
                results.append(out)
    else: # sequential execution
        _init_worker(seq_len, hist_scaler,
                     neg_per_pos,
                     SEED,
                     NEG_HISTORY_BACKSHIFT_MIN, NEG_HISTORY_BACKSHIFT_MAX,
                     mean_time_arr)
        for a in tqdm(args_iter, desc=desc):
            results.append(_process_one_player(a))

    H_lv, H_ct, H_ln, T_id, T_mean, Y, G = [], [], [], [], [], [], []
    for r in results: 
        if r[0].shape[0] == 0:
            continue
        H_lv.append(r[0]); H_ct.append(r[1]); H_ln.append(r[2])
        T_id.append(r[3]); T_mean.append(r[4]); Y.append(r[5]); G.append(r[6])

    if len(Y) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    hist_levels = np.concatenate(H_lv, axis=0)
    hist_cont   = np.concatenate(H_ct, axis=0) 
    hist_len    = np.concatenate(H_ln, axis=0)
    tgt_ids     = np.concatenate(T_id, axis=0)
    tgt_means   = np.concatenate(T_mean, axis=0)
    labels      = np.concatenate(Y, axis=0)
    groups      = np.concatenate(G, axis=0)

    return hist_levels, hist_cont, hist_len, tgt_ids, tgt_means, labels, groups

# =====================
# 3) Flatten & concat
# =====================

def flatten_and_concat(hist_cont: np.ndarray,
                       tgt_ids: np.ndarray,
                       tgt_means: np.ndarray, 
                       hist_len: np.ndarray,
                       use_hist_len: bool = True) -> Tuple[np.ndarray, List[str]]:
    N, T, F = hist_cont.shape 
    X_hist = hist_cont.reshape(N, T * F)
    
    X_tgt_id = tgt_ids.reshape(-1, 1).astype(np.float32)
    X_tgt_mean = tgt_means.reshape(-1, 1).astype(np.float32)
    
    X_parts = [X_hist, X_tgt_id, X_tgt_mean]
    
    feat_names = [f"h_{{t}}_{f}" for t in range(T) for f in range(F)] 
    feat_names.append("target_level_id")
    feat_names.append("target_mean_time")
    
    if use_hist_len:
        X_parts.append(hist_len.reshape(-1, 1).astype(np.float32))
        feat_names.append("hist_len")
        
    X = np.concatenate(X_parts, axis=1).astype(np.float32)
    return X, feat_names

def save_subset(prefix, hist_levels, hist_cont, hist_len, tgt_ids, tgt_means, labels, groups):
    if len(labels) == 0:
        print(f"Warning: {prefix} subset is empty!")
        return None

    X_all, feat_names = flatten_and_concat(hist_cont, tgt_ids, tgt_means, hist_len, use_hist_len=True)
    y_all = labels.astype(np.float32)
    g_all = groups.astype(np.int64)

    path_x = os.path.join(OUTPUT_DIR, f"X_{prefix}.npy")
    path_y = os.path.join(OUTPUT_DIR, f"y_{prefix}.npy")
    path_g = os.path.join(OUTPUT_DIR, f"groups_{prefix}.npy")
    
    np.save(path_x, X_all)
    np.save(path_y, y_all)
    np.save(path_g, g_all)
    
    return feat_names

# =====================
# 4) Main
# =====================
def main():
    assert os.path.exists(INPUT_CSV_PATH), f"File does not exist: {INPUT_CSV_PATH}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_players = load_players_from_csv(INPUT_CSV_PATH)

    random.shuffle(all_players)
    
    n_total = len(all_players)
    n_train = int(n_total * TRAIN_SPLIT_RATIO)
    
    train_players = all_players[:n_train]
    holdout_players = all_players[n_train:]
    
    hist_scaler, _ = compute_hist_scaler(all_players)

    # Build Train Set
    print(f"\nBuilding TRAIN set (Ratio 1 Pos : {TRAIN_NEG_PER_POS} Neg)...")
    train_data = build_dataset_fast(
         train_players, SEQ_LEN, hist_scaler,
         neg_per_pos=TRAIN_NEG_PER_POS,
         n_workers=N_WORKERS,
         mean_time_arr=mean_time
     )
    
    feat_names = save_subset("train", *train_data)

    # Build Holdout Set
    print(f"\nBuilding HOLDOUT set (ALL Negatives - Extreme Imbalance)...")
    holdout_data = build_dataset_fast(
         holdout_players, SEQ_LEN, hist_scaler,
         neg_per_pos=None,
         n_workers=N_WORKERS,
         mean_time_arr=mean_time
     )
    
    save_subset("holdout", *holdout_data)

    if feat_names:
        with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
            json.dump(feat_names, f, ensure_ascii=False, indent=2)

    print("\n=== All data prepared successfully ===")

if __name__ == "__main__":
    main()
