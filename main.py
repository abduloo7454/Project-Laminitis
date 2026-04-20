# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================
# !pip install -q eli5 imbalanced-learn tabpfn xgboost

# ============================================================
# IMPORTS
# ============================================================
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from tabpfn import TabPFNClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import eli5
import warnings
from math import ceil
from itertools import cycle
warnings.filterwarnings("ignore")

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost loaded successfully")
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost not available: {e}")

# TabPFN
try:
    os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYjQxZjU3ZWQtOTU2ZC00ZGE3LWI4ZGUtODEyOTU1MTFhY2U2IiwiZXhwIjoxODA3MTgzMTcyfQ.hulZOnyiEDXda879s_8cjcDu7MG9ugKHmep5JieKJ38"
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("TabPFN loaded successfully")
except Exception as e:
    TABPFN_AVAILABLE = False
    print(f"TabPFN not available: {e}")

RNG            = 42
CORR_THRESHOLD = 0.7
N_FEATURES_SEARCH = [10, 12, 15, 18, 20]

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(
    'preprocessed.xlsx'
)
X = df.drop(columns=['Class']).apply(
    pd.to_numeric, errors='coerce'
).fillna(0)
y             = df['Class'].values
n_classes     = len(np.unique(y))
labels_global = np.unique(y)

print(
    f"Dataset: {X.shape[0]} horses, "
    f"{X.shape[1]} features"
)
print(
    f"Class distribution: "
    f"{dict(zip(*np.unique(y, return_counts=True)))}"
)

# ============================================================
# DEFINE BASE MODELS
# ============================================================
models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=RNG
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight='balanced',
            random_state=RNG
        ))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight='balanced',
        random_state=RNG,
        n_jobs=-1
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=None,
        class_weight='balanced',
        random_state=RNG
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=RNG
    ),
    "ANN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=RNG
        ))
    ])
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RNG,
        n_jobs=-1,
        verbosity=0
    )
    print("XGBoost added to models")

# TabPFN
try:
    import os
    os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYjQxZjU3ZWQtOTU2ZC00ZGE3LWI4ZGUtODEyOTU1MTFhY2U2IiwiZXhwIjoxODA3MTgzMTcyfQ.hulZOnyiEDXda879s_8cjcDu7MG9ugKHmep5JieKJ38"
    from tabpfn import TabPFNClassifier
    # Test that token works by creating instance
    _ = TabPFNClassifier(
        random_state=RNG,
        ignore_pretraining_limits=True
    )
    TABPFN_AVAILABLE = True
    print("TabPFN loaded and authenticated successfully")
except Exception as e:
    TABPFN_AVAILABLE = False
    print(f"TabPFN not available: {e}")

# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================
param_grids = {
    "LogReg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0]
    },
    "SVM": {
        "clf__C"    : [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto"]
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 400],
        "max_depth"   : [None, 5, 10]
    },
    "DecisionTree": {
        "max_depth": [None, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators" : [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1]
    },
    "ANN": {
        "clf__hidden_layer_sizes": [
            (64, 32), (128, 64), (32,)
        ]
    }
}

if XGBOOST_AVAILABLE:
    param_grids["XGBoost"] = {
        "n_estimators" : [100, 200, 400],
        "max_depth"    : [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }

if TABPFN_AVAILABLE:
    import torch
    tabpfn_device = "mps" if torch.backends.mps.is_available() else "auto"
    models["TabPFN"] = TabPFNClassifier(
        random_state=RNG,
        ignore_pretraining_limits=True,
        device=tabpfn_device
    )
    param_grids["TabPFN"] = {}
    print("TabPFN added to models")

# Drop limb-specific columns before CV
X = X.drop(columns=['LERF', 'LELF', 'LERH', 'LELH'], errors='ignore')

# ============================================================
# NESTED 5-FOLD CROSS-VALIDATION
# Outer loop: 10-fold CV on whole dataset
# Inner loop: 10-fold CV within each training fold
#   - searches over number of features
#   - tunes hyperparameters
#   - applies SMOTE on training data only
# No data leakage at any stage
# ============================================================
outer_cv = StratifiedKFold(
    n_splits=10, shuffle=True, random_state=RNG
)
inner_cv = StratifiedKFold(
    n_splits=10, shuffle=True, random_state=RNG
)

fold_results     = []
roc_data         = {}
conf_matrices    = {}
best_model_name  = None
best_model_score = -np.inf

# Store best configuration per model for
# final fitting on whole dataset
best_configs = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    fold_metrics      = []
    y_true_all        = []
    y_pred_all        = []
    y_proba_all       = []
    fold_best_configs = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # Step 1: Correlation removal on
        # training fold only
        corr_matrix = X_tr.corr().abs()
        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape), k=1
            ).astype(bool)
        )
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > CORR_THRESHOLD)
        ]
        X_tr = X_tr.drop(
            columns=to_drop, errors='ignore'
        )
        X_te = X_te.drop(
            columns=to_drop, errors='ignore'
        )

        # Step 2: Compute feature importances
        # on training fold only
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            random_state=RNG,
            n_jobs=-1
        )
        rf_selector.fit(X_tr, y_tr)
        importances = pd.Series(
            rf_selector.feature_importances_,
            index=X_tr.columns
        ).sort_values(ascending=False)

        # Step 3: Inner loop
        # Search over number of features
        # AND hyperparameters simultaneously
        best_inner_score  = -np.inf
        best_n_features   = N_FEATURES_SEARCH[0]
        best_inner_params = {}
        param_grid        = param_grids.get(name, {})

        # TabPFN is a pre-trained model — skip inner CV
        # and use fixed top N features to avoid 125 fits
        if name == "TabPFN":
            best_n_features = max(N_FEATURES_SEARCH)
            best_inner_score = 0.0
            best_inner_params = {}
            top_features = importances.head(
                best_n_features
            ).index.tolist()
            X_tr = X_tr[top_features]
            X_te = X_te[top_features]
            smote_skip = SMOTE(random_state=RNG)
            X_tr_res, y_tr_res = smote_skip.fit_resample(
                X_tr, y_tr
            )
            fold_model = clone(model)
            fold_model.fit(X_tr_res, y_tr_res)
            y_pred = fold_model.predict(X_te)
            if hasattr(fold_model, "predict_proba"):
                y_proba = fold_model.predict_proba(X_te)
            else:
                y_proba = np.zeros((len(y_pred), n_classes))
            avg = "binary" if n_classes == 2 else "macro"
            acc  = accuracy_score(y_te, y_pred)
            prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
            rec  = recall_score(y_te, y_pred, average=avg, zero_division=0)
            f1   = f1_score(y_te, y_pred, average=avg, zero_division=0)
            mcc  = matthews_corrcoef(y_te, y_pred)
            auc  = roc_auc_score(y_te, y_proba[:, 1]) if n_classes == 2 else roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro')
            fold_metrics.append({
                "Model"     : name,
                "Fold"      : fold,
                "N_Features": best_n_features,
                "Train_Acc" : None, "Train_F1": None, "Train_MCC": None,
                "Val_F1"    : None,
                "Test_Acc"  : round(acc, 3), "Test_Prec": round(prec, 3),
                "Test_Rec"  : round(rec, 3), "Test_F1": round(f1, 3),
                "Test_MCC"  : round(mcc, 3), "Test_AUC": round(auc, 3)
            })
            y_true_all.extend(y_te)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)
            fold_best_configs.append({
                "fold"        : fold,
                "n_features"  : best_n_features,
                "params"      : {},
                "val_f1"      : None,
                "top_features": top_features
            })
            print(f"\n  Fold {fold}:")
            print(f"    Test — Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} MCC={mcc:.3f} AUC={auc:.3f}")
            continue

        for n_feat in N_FEATURES_SEARCH:
            top_feats = importances.head(n_feat).index.tolist()
            X_tr_n    = X_tr[top_feats]  # original — SMOTE inside each fold

            if param_grid:
                # ImbPipeline keeps SMOTE inside each inner fold (no leakage)
                if hasattr(clone(model), 'steps'):
                    imb_model     = ImbPipeline(
                        [('smote', SMOTE(random_state=RNG))]
                        + list(clone(model).steps)
                    )
                    gs_param_grid = param_grid
                else:
                    imb_model     = ImbPipeline([
                        ('smote', SMOTE(random_state=RNG)),
                        ('clf',   clone(model))
                    ])
                    gs_param_grid = {
                        f'clf__{k}': v for k, v in param_grid.items()
                    }
                gs = GridSearchCV(
                    estimator=imb_model,
                    param_grid=gs_param_grid,
                    cv=inner_cv,
                    scoring='f1',
                    n_jobs=-1,
                    refit=True
                )
                gs.fit(X_tr_n, y_tr)
                score = gs.best_score_
                if hasattr(clone(model), 'steps'):
                    params = gs.best_params_
                else:
                    params = {
                        k[5:]: v
                        for k, v in gs.best_params_.items()
                        if k.startswith('clf__')
                    }
            else:
                # No hyperparameters — SMOTE inside each inner fold manually
                inner_scores = []
                for inn_tr, inn_val in inner_cv.split(X_tr_n, y_tr):
                    smote_inn = SMOTE(random_state=RNG)
                    X_inn_res, y_inn_res = smote_inn.fit_resample(
                        X_tr_n.iloc[inn_tr], y_tr[inn_tr]
                    )
                    inn_model = clone(model)
                    inn_model.fit(X_inn_res, y_inn_res)
                    inn_pred = inn_model.predict(X_tr_n.iloc[inn_val])
                    inner_scores.append(f1_score(
                        y_tr[inn_val], inn_pred,
                        average='binary', zero_division=0
                    ))
                score  = np.mean(inner_scores)
                params = {}

            if score > best_inner_score:
                best_inner_score  = score
                best_n_features   = n_feat
                best_inner_params = params

        print(
            f"\n  Fold {fold} — "
            f"Best n_features: {best_n_features} | "
            f"Best val F1: {best_inner_score:.3f} | "
            f"Best params: {best_inner_params}"
        )

        # Store fold best config
        fold_best_configs.append({
            "fold"        : fold,
            "n_features"  : best_n_features,
            "params"      : best_inner_params,
            "val_f1"      : best_inner_score,
            "top_features": importances.head(
                best_n_features
            ).index.tolist()
        })

        # Step 4: Retrain with best configuration
        # on full outer training fold
        best_top_features = importances.head(
            best_n_features
        ).index.tolist()
        X_tr_best = X_tr[best_top_features]
        X_te_best = X_te[best_top_features]

        smote = SMOTE(random_state=RNG)
        X_tr_res, y_tr_res = smote.fit_resample(
            X_tr_best, y_tr
        )
        X_tr_res_df = pd.DataFrame(
            X_tr_res, columns=best_top_features
        )

        fold_model = clone(model)
        if best_inner_params:
            fold_model.set_params(**best_inner_params)
        fold_model.fit(X_tr_res_df, y_tr_res)

        # ---- Training metrics ----
        y_pred_train = fold_model.predict(X_tr_res_df)
        avg_mode     = (
            "binary" if n_classes == 2 else "macro"
        )
        train_acc = accuracy_score(
            y_tr_res, y_pred_train
        )
        train_f1  = f1_score(
            y_tr_res, y_pred_train,
            average=avg_mode, zero_division=0
        )
        train_mcc = matthews_corrcoef(
            y_tr_res, y_pred_train
        )

        # ---- Test metrics ----
        y_pred = fold_model.predict(X_te_best)

        if hasattr(fold_model, "predict_proba"):
            y_proba = fold_model.predict_proba(
                X_te_best
            )
        else:
            y_proba = np.zeros(
                (len(y_pred), n_classes)
            )

        avg  = "binary" if n_classes == 2 else "macro"
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(
            y_te, y_pred,
            average=avg, zero_division=0
        )
        rec  = recall_score(
            y_te, y_pred,
            average=avg, zero_division=0
        )
        f1   = f1_score(
            y_te, y_pred,
            average=avg, zero_division=0
        )
        mcc  = matthews_corrcoef(y_te, y_pred)
        if n_classes == 2:
            auc = roc_auc_score(
                y_te, y_proba[:, 1]
            )
        else:
            auc = roc_auc_score(
                y_te, y_proba,
                multi_class='ovr',
                average='macro'
            )

        fold_metrics.append({
            "Model"      : name,
            "Fold"       : fold,
            "N_Features" : best_n_features,
            "Train_Acc"  : round(train_acc,        3),
            "Train_F1"   : round(train_f1,         3),
            "Train_MCC"  : round(train_mcc,        3),
            "Val_F1"     : round(best_inner_score,  3),
            "Test_Acc"   : round(acc,              3),
            "Test_Prec"  : round(prec,             3),
            "Test_Rec"   : round(rec,              3),
            "Test_F1"    : round(f1,               3),
            "Test_MCC"   : round(mcc,              3),
            "Test_AUC"   : round(auc,              3)
        })

        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)

        print(f"  Fold {fold}:")
        print(
            f"    Training   — "
            f"Acc={train_acc:.3f} "
            f"F1={train_f1:.3f} "
            f"MCC={train_mcc:.3f}"
        )
        print(
            f"    Validation — "
            f"Best Val F1={best_inner_score:.3f} "
            f"N_Features={best_n_features}"
        )
        print(
            f"    Test       — "
            f"Acc={acc:.3f} "
            f"Prec={prec:.3f} "
            f"Rec={rec:.3f} "
            f"F1={f1:.3f} "
            f"MCC={mcc:.3f} "
            f"AUC={auc:.3f}"
        )

    # Average metrics across folds
    fold_df = pd.DataFrame(fold_metrics)
    fold_results.append(fold_df)
    avg_test_f1 = fold_df["Test_F1"].mean()

    print(f"\n  {'─'*50}")
    print(f"  Average across 5 folds — {name}")
    print(f"  {'─'*50}")
    for col, label in [
        ("N_Features", "N Features"),
        ("Train_Acc",  "Train Acc "),
        ("Train_F1",   "Train F1  "),
        ("Train_MCC",  "Train MCC "),
        ("Val_F1",     "Val   F1  "),
        ("Test_Acc",   "Test  Acc "),
        ("Test_Prec",  "Test  Prec"),
        ("Test_Rec",   "Test  Rec "),
        ("Test_F1",    "Test  F1  "),
        ("Test_MCC",   "Test  MCC "),
        ("Test_AUC",   "Test  AUC "),
    ]:
        col_mean = pd.to_numeric(fold_df[col], errors='coerce').mean()
        col_std  = pd.to_numeric(fold_df[col], errors='coerce').std()
        if pd.isna(col_mean):
            print(f"  {label} = N/A")
        else:
            print(
                f"  {label} = "
                f"{col_mean:.3f}"
                f" +/- {col_std:.3f}"
            )

    # Store best configs for this model
    best_configs[name] = fold_best_configs

    # Track best model across all models
    if avg_test_f1 > best_model_score:
        best_model_score = avg_test_f1
        best_model_name  = name

    # Aggregate confusion matrix
    y_true_arr  = np.array(y_true_all)
    y_pred_arr  = np.array(y_pred_all)
    y_proba_arr = np.array(y_proba_all)

    cm = confusion_matrix(
        y_true_arr, y_pred_arr,
        labels=labels_global
    )
    row_sums = cm.sum(
        axis=1, keepdims=True
    ).clip(min=1)
    conf_matrices[name] = (
        cm.astype(float) / row_sums
    )

    # ROC curve
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(
            y_true_arr, y_proba_arr[:, 1]
        )
        auc_score = roc_auc_score(
            y_true_arr, y_proba_arr[:, 1]
        )
        roc_data[name] = (fpr, tpr, auc_score)

# ============================================================
# SUMMARY TABLES
# ============================================================
all_folds_df = pd.concat(
    fold_results, ignore_index=True
)

print("\n\n===== PER-FOLD RESULTS =====")
print(all_folds_df.to_string(index=False))

print("\n\n===== AVERAGE RESULTS =====")
summary = all_folds_df.groupby("Model").agg(
    N_Feat_mean    =("N_Features", "mean"),
    N_Feat_std     =("N_Features", "std"),
    Train_Acc_mean =("Train_Acc",  "mean"),
    Train_Acc_std  =("Train_Acc",  "std"),
    Train_F1_mean  =("Train_F1",   "mean"),
    Train_F1_std   =("Train_F1",   "std"),
    Train_MCC_mean =("Train_MCC",  "mean"),
    Train_MCC_std  =("Train_MCC",  "std"),
    Val_F1_mean    =("Val_F1",     "mean"),
    Val_F1_std     =("Val_F1",     "std"),
    Test_Acc_mean  =("Test_Acc",   "mean"),
    Test_Acc_std   =("Test_Acc",   "std"),
    Test_Prec_mean =("Test_Prec",  "mean"),
    Test_Prec_std  =("Test_Prec",  "std"),
    Test_Rec_mean  =("Test_Rec",   "mean"),
    Test_Rec_std   =("Test_Rec",   "std"),
    Test_F1_mean   =("Test_F1",    "mean"),
    Test_F1_std    =("Test_F1",    "std"),
    Test_MCC_mean  =("Test_MCC",   "mean"),
    Test_MCC_std   =("Test_MCC",   "std"),
    Test_AUC_mean  =("Test_AUC",   "mean"),
    Test_AUC_std   =("Test_AUC",   "std"),
).reset_index()
print(summary.to_string(index=False))

all_folds_df.to_excel(
    "fold_results.xlsx", index=False
)
summary.to_excel(
    "average_results.xlsx", index=False
)
print(
    "\nResults saved to fold_results.xlsx"
    " and average_results.xlsx"
)

# ============================================================
# FIT BEST PIPELINE ON WHOLE DATASET
# "Take the best performing pipeline and fit it
# to the whole dataset and analyze its selected
# features and coefficients"
# ============================================================
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(
    f"Average Test F1: {best_model_score:.3f}"
)
print(f"{'='*60}")

# Determine the most frequently selected
# number of features across folds
fold_n_features = [
    cfg["n_features"]
    for cfg in best_configs[best_model_name]
]
final_n_features = int(
    pd.Series(fold_n_features).mode()[0]
)
print(
    f"\nMost frequently selected number of "
    f"features across folds: {final_n_features}"
)

# Determine most frequent hyperparameters
# across folds
fold_params = [
    cfg["params"]
    for cfg in best_configs[best_model_name]
    if cfg["params"]
]
if fold_params:
    params_df    = pd.DataFrame(fold_params)
    final_params = {
        col: params_df[col].mode()[0]
        for col in params_df.columns
    }
    print(
        f"Most frequent hyperparameters: "
        f"{final_params}"
    )
else:
    final_params = {}

# Step 1: Correlation removal on full dataset
print(
    "\nStep 1: Removing correlated features "
    "on full dataset..."
)
corr_full  = X.corr().abs()
upper_full = corr_full.where(
    np.triu(
        np.ones(corr_full.shape), k=1
    ).astype(bool)
)
to_drop_full = [
    col for col in upper_full.columns
    if any(upper_full[col] > CORR_THRESHOLD)
]
X_full = X.drop(
    columns=to_drop_full, errors='ignore'
)
print(
    f"  Dropped {len(to_drop_full)} correlated "
    f"features. Remaining: {X_full.shape[1]}"
)

# Step 2: Feature selection on full dataset
print(
    "\nStep 2: Selecting top features "
    "on full dataset..."
)
rf_full = RandomForestClassifier(
    n_estimators=100, random_state=RNG, n_jobs=-1
)
rf_full.fit(X_full, y)
importances_full = pd.Series(
    rf_full.feature_importances_,
    index=X_full.columns
).sort_values(ascending=False)
final_top_features = importances_full.head(
    final_n_features
).index.tolist()
X_final       = X_full[final_top_features]
feature_names = X_final.columns.tolist()

print(f"\nFinal selected features ({final_n_features}):")
for i, f in enumerate(feature_names, 1):
    print(
        f"  {i:2d}. {f:40s} "
        f"importance={importances_full[f]:.4f}"
    )

# Step 3: SMOTE on full dataset
print(
    "\nStep 3: Applying SMOTE on full dataset..."
)
smote_full = SMOTE(random_state=RNG)
X_res_full, y_res_full = smote_full.fit_resample(
    X_final, y
)
X_res_full_df = pd.DataFrame(
    X_res_full, columns=feature_names
)
print(
    f"  After SMOTE: {X_res_full_df.shape[0]} "
    f"samples"
)

# Step 4: Hyperparameter tuning on full dataset
# using inner CV
print(
    "\nStep 4: Tuning hyperparameters on "
    "full dataset..."
)
best_base_model = models[best_model_name]
best_param_grid = param_grids.get(
    best_model_name, {}
)
inner_cv_full = StratifiedKFold(
    n_splits=10, shuffle=True, random_state=RNG
)

if best_param_grid:
    final_grid = GridSearchCV(
        estimator=clone(best_base_model),
        param_grid=best_param_grid,
        cv=inner_cv_full,
        scoring='f1',
        n_jobs=-1,
        refit=True
    )
    final_grid.fit(X_res_full_df, y_res_full)
    best_pipeline       = final_grid.best_estimator_
    final_best_params   = final_grid.best_params_
    print(
        f"  Final best hyperparameters: "
        f"{final_best_params}"
    )
else:
    best_pipeline = clone(best_base_model)
    best_pipeline.fit(X_res_full_df, y_res_full)
    final_best_params = {}
    print("  No hyperparameters to tune.")

print(
    f"\nBest pipeline ({best_model_name}) fitted "
    f"on full dataset successfully."
)

# ============================================================
# ANALYZE SELECTED FEATURES AND COEFFICIENTS
# and coefficients"
# ============================================================
print(f"\n{'='*60}")
print("FEATURE ANALYSIS — BEST PIPELINE")
print(f"{'='*60}")

# Feature importance analysis
feature_importance_df = pd.DataFrame({
    "Feature"   : feature_names,
    "Importance": [
        importances_full[f] for f in feature_names
    ]
}).sort_values("Importance", ascending=False)

print("\nFeature importances (from Random Forest "
      "selection on full dataset):")
print(feature_importance_df.to_string(index=False))

# Extract coefficients if model supports it
print(
    f"\nCoefficient analysis for "
    f"{best_model_name}:"
)

try:
    if hasattr(best_pipeline, 'coef_'):
        # Linear models: LogReg, SVM linear
        coefs = best_pipeline.coef_[0]
        coef_df = pd.DataFrame({
            "Feature"    : feature_names,
            "Coefficient": coefs,
            "Odds_Ratio" : np.exp(coefs)
        }).sort_values(
            "Coefficient",
            ascending=False,
            key=abs
        )
        print(coef_df.to_string(index=False))
        coef_df.to_excel(
            "best_model_coefficients.xlsx",
            index=False
        )
        print(
            "\nCoefficients saved to "
            "best_model_coefficients.xlsx"
        )

    elif hasattr(best_pipeline, 'feature_importances_'):
        # Tree-based models: RF, DT, GB, XGBoost
        feat_imp = pd.DataFrame({
            "Feature"   : feature_names,
            "Importance": (
                best_pipeline.feature_importances_
            )
        }).sort_values(
            "Importance", ascending=False
        )
        print(feat_imp.to_string(index=False))
        feat_imp.to_excel(
            "best_model_feature_importances.xlsx",
            index=False
        )
        print(
            "\nFeature importances saved to "
            "best_model_feature_importances.xlsx"
        )

    elif hasattr(best_pipeline, 'named_steps'):
        # Pipeline models: check inner classifier
        clf = best_pipeline.named_steps.get('clf')
        if clf is not None:
            if hasattr(clf, 'coef_'):
                coefs = clf.coef_[0]
                coef_df = pd.DataFrame({
                    "Feature"    : feature_names,
                    "Coefficient": coefs,
                    "Odds_Ratio" : np.exp(coefs)
                }).sort_values(
                    "Coefficient",
                    ascending=False,
                    key=abs
                )
                print(coef_df.to_string(index=False))
                coef_df.to_excel(
                    "best_model_coefficients.xlsx",
                    index=False
                )
            elif hasattr(clf, 'feature_importances_'):
                feat_imp = pd.DataFrame({
                    "Feature"   : feature_names,
                    "Importance": (
                        clf.feature_importances_
                    )
                }).sort_values(
                    "Importance", ascending=False
                )
                print(feat_imp.to_string(index=False))
                feat_imp.to_excel(
                    "best_model_feature_importances"
                    ".xlsx",
                    index=False
                )
    else:
        print(
            "  Model does not expose coefficients "
            "or feature importances directly."
        )

except Exception as e:
    print(f"  Could not extract coefficients: {e}")

# ============================================================
# ELI5 FEATURE IMPORTANCE
# Applied on full dataset after CV is complete
# No SMOTE — class imbalance handled via class_weight
# TabPFN excluded — incompatible with ELI5
# ============================================================
print("\n\n===== ELI5 FEATURE IMPORTANCE =====")
print(
    "Note: ELI5 applied to classifiers supporting "
    "internal weight extraction. TabPFN excluded."
)

eli5_models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced',
        random_state=RNG
    ),
    "LogReg": LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=RNG
    ),
    "SVM": SVC(
        kernel="linear",
        class_weight='balanced',
        random_state=RNG
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=None,
        class_weight='balanced',
        random_state=RNG
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        random_state=RNG
    ),
}

if XGBOOST_AVAILABLE:
    eli5_models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        random_state=RNG,
        n_jobs=-1,
        verbosity=0
    )

eli5_results = {}
for name, eli5_model in eli5_models.items():
    print(f"\n----- ELI5: {name} -----")
    eli5_model.fit(X_final, y)
    weights   = eli5.explain_weights(
        eli5_model,
        feature_names=feature_names
    )
    weight_df = eli5.format_as_dataframe(weights)
    print(weight_df.to_string(index=False))
    eli5_results[name] = weight_df

with pd.ExcelWriter("eli5_results.xlsx") as writer:
    for name, df_w in eli5_results.items():
        df_w.to_excel(
            writer, sheet_name=name, index=False
        )
print("\nELI5 results saved to eli5_results.xlsx")

# ============================================================
# CONFUSION MATRICES PLOT
# ============================================================
num_models = len(conf_matrices)
cols       = 3
rows       = ceil(num_models / cols)

fig, axes = plt.subplots(
    rows, cols,
    figsize=(6*cols, 6*rows),
    dpi=300
)
axes = axes.ravel()

for i, (name, cm_norm) in enumerate(
    conf_matrices.items()
):
    cm_pct = cm_norm * 100.0
    disp   = ConfusionMatrixDisplay(
        confusion_matrix=cm_pct,
        display_labels=labels_global
    )
    disp.plot(
        ax=axes[i],
        cmap='viridis',
        colorbar=False,
        values_format=".0f"
    )
    axes[i].set_title(
        name, fontsize=16, weight='bold'
    )
    axes[i].set_xlabel(
        "Predicted label", fontsize=12
    )
    axes[i].set_ylabel(
        "True label", fontsize=12
    )
    for txt in axes[i].texts:
        txt.set_text(f"{txt.get_text()}%")
        txt.set_fontsize(20)
        txt.set_weight('bold')

for j in range(i+1, len(axes)):
    axes[j].axis('off')

fig.suptitle(
    "Mean Confusion Matrices — "
    "5-Fold CV — All Models",
    fontsize=18, weight='bold'
)
plt.tight_layout()
plt.savefig(
    "confusion_matrices.pdf",
    format='pdf', dpi=300
)
plt.show()
print(
    "Confusion matrices saved to "
    "confusion_matrices.pdf"
)

# ============================================================
# ROC CURVES PLOT
# ============================================================
colors = cycle([
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22"
])

plt.figure(figsize=(8, 7), dpi=300)
for (name, (fpr, tpr, auc_val)), color in zip(
    roc_data.items(), colors
):
    plt.plot(
        fpr, tpr, color=color, lw=2.5,
        label=f"{name} (AUC = {auc_val:.3f})"
    )
plt.plot([0,1],[0,1],'k--', lw=1.2, alpha=0.6)
plt.title(
    "ROC Curves — All Models",
    fontsize=14, weight='bold'
)
plt.xlabel("False Positive Rate", fontsize=13)
plt.ylabel("True Positive Rate",  fontsize=13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(
    True, linestyle="--",
    linewidth=0.6, alpha=0.4
)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(
    "roc_curves.pdf", format='pdf', dpi=300
)
plt.show()
print("ROC curves saved to roc_curves.pdf")

# ============================================================
# CORRELATION MATRIX PLOT (-1 to 1)
# ============================================================
plt.figure(figsize=(16, 14))
sns.heatmap(
    X.corr(),
    annot=True,
    fmt=".2f",
    linewidths=0.8,
    cmap='coolwarm',
    annot_kws={"size": 8},
    vmin=-1,
    vmax=1,
    center=0
)
plt.title('Correlation Matrix', fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    "correlation_matrix.pdf",
    format='pdf', dpi=300
)
plt.show()
print(
    "Correlation matrix saved to "
    "correlation_matrix.pdf"
)

print("\n\n===== ALL DONE =====")
print(
    f"Best model: {best_model_name} "
    f"(Avg Test F1 = {best_model_score:.3f})"
)
print(
    f"Final selected features: {feature_names}"
)
print(
    f"Final hyperparameters: {final_best_params}"
)