"""
AIQUAM++ — Script di training
Legge i campioni CSV pre-costruiti da dataset/<anno>/ e addestra i 4 modelli:
  - KNN + DTW   (knn_dtw.KnnDtw)
  - KNN sklearn (KNeighborsClassifier)
  - CNN         (3-branch multi-scale)
  - ResNet 1D   (3 blocchi residuali)

Tutti i parametri configurabili sono in config.ini / config.py.
Le classi KnnDtw e ProgressBar sono in knn_dtw.py.
"""

import collections
import glob
import os
import pickle

import matplotlib
matplotlib.use('Agg')   # headless; cambia in 'TkAgg' per plot interattivi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from config import cfg_training as cfg
from knn_dtw import KnnDtw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true, y_pred, title):
    conf_mat = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(7, 7))
    res = plt.imshow(conf_mat, cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j - .2, i + .1, c, fontsize=16)
    fig.colorbar(res)
    plt.title('Confusion Matrix — ' + title)
    plt.xticks(range(cfg.n_classes), list(cfg.labels.values()), rotation=90)
    plt.yticks(range(cfg.n_classes), list(cfg.labels.values()))
    path = os.path.join(cfg.models_dir, f"confusion_{title.replace(' ', '_')}.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  → confusion matrix salvata in {path}")


def plot_training_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title(f'{title} — accuracy')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('accuracy'); ax1.legend()

    ax2.plot(history.history['loss'],     label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title(f'{title} — loss')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('loss'); ax2.legend()

    path = os.path.join(cfg.models_dir, f"history_{title.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  → curva di training salvata in {path}")


# ---------------------------------------------------------------------------
# 1. Caricamento dataset
# ---------------------------------------------------------------------------

print("=" * 60)
print("Caricamento dataset …")
print("=" * 60)

os.makedirs(cfg.models_dir, exist_ok=True)

all_features, all_labels, skipped = [], [], []

for year in cfg.dataset_years:
    pattern = os.path.join(cfg.dataset_dir, year, "*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  [WARN] Nessun CSV in {os.path.join(cfg.dataset_dir, year)}")
        continue

    for fpath in files:
        if fpath.endswith("_matrix.csv"):
            continue
        try:
            df = pd.read_csv(fpath)

            missing = [c for c in cfg.feature_cols if c not in df.columns]
            if missing:
                skipped.append((fpath, f"colonne mancanti: {missing}"))
                continue
            if "target" not in df.columns:
                skipped.append((fpath, "colonna 'target' mancante"))
                continue

            features = df[cfg.feature_cols].values.astype(np.float32)
            label    = df["target"].values.astype(np.int32)

            valid = ~(np.isnan(features).any(axis=1) | np.isnan(label))
            features, label = features[valid], label[valid]

            if len(features) == 0:
                skipped.append((fpath, "nessuna riga valida dopo filtro NaN"))
                continue

            all_features.append(features)
            all_labels.append(label)

        except Exception as e:
            skipped.append((fpath, str(e)))

if not all_features:
    raise RuntimeError(
        f"Nessun campione caricato da '{cfg.dataset_dir}'. "
        "Verifica che la directory esista e contenga file CSV (non _matrix)."
    )

X = np.vstack(all_features)
y = np.concatenate(all_labels)

print(f"\nCaricati : {X.shape[0]} campioni  |  {X.shape[1]} features")
print(f"Classi   : {dict(sorted(collections.Counter(y.tolist()).items()))}")
if skipped:
    print(f"\nSaltati {len(skipped)} file:")
    for path, reason in skipped:
        print(f"  {os.path.basename(path)}  →  {reason}")

# ---------------------------------------------------------------------------
# 2. SMOTE
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SMOTE …")
print("=" * 60)

counter      = collections.Counter(y.tolist())
y_0          = counter.get(0, 0)
majority_n   = max(counter.values())

smote_strategy = {}
for cls in [1, 2]:
    current  = counter.get(cls, 0)
    target_n = int(y_0 / 2) if y_0 > 0 else int(majority_n / 2)
    if target_n > current:
        smote_strategy[cls] = target_n

if smote_strategy:
    print(f"  Distribuzione prima : {dict(sorted(counter.items()))}")
    print(f"  Target SMOTE        : {smote_strategy}")
    min_minority = min(counter.get(c, 0) for c in smote_strategy)
    k_neighbors  = min(5, min_minority - 1) if min_minority > 1 else 1
    smt = SMOTE(sampling_strategy=smote_strategy,
                k_neighbors=k_neighbors,
                random_state=cfg.random_seed)
    X, y = smt.fit_resample(X, y)
    print(f"  Distribuzione dopo  : {dict(sorted(collections.Counter(y.tolist()).items()))}")
else:
    print("  Classi già bilanciate — SMOTE saltato.")

# ---------------------------------------------------------------------------
# 3. Split stratificato
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg.test_size,
    stratify=y,
    random_state=cfg.random_seed
)
n_classes = len(np.unique(y_train))

print(f"\nTrain : {X_train.shape[0]} campioni")
print(f"Test  : {X_test.shape[0]}  campioni")

# ---------------------------------------------------------------------------
# 4a. KNN + DTW
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training KNN + DTW …")
print("=" * 60)

model_knn_dtw = KnnDtw(
    n_neighbors=cfg.knn_dtw_n_neighbors,
    max_warping_window=cfg.knn_dtw_warping_window
)
model_knn_dtw.fit(X_train, y_train)

knn_dtw_path = os.path.join(cfg.models_dir, "KNN_DTW.pkl")
with open(knn_dtw_path, "wb") as f:
    pickle.dump(model_knn_dtw, f)
print(f"  → salvato in {knn_dtw_path}")

y_pred_dtw, _ = model_knn_dtw.predict(X_test)
print("\nKNN + DTW")
print(classification_report(y_test, y_pred_dtw,
                             target_names=list(cfg.labels.values()), zero_division=0))
print_confusion_matrix(y_test, y_pred_dtw, "KNN DTW")

# ---------------------------------------------------------------------------
# 4b. KNN sklearn
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training KNN sklearn …")
print("=" * 60)

model_knn = KNeighborsClassifier(n_neighbors=cfg.knn_n_neighbors)
model_knn.fit(X_train, y_train)

knn_path = os.path.join(cfg.models_dir, "KNeighborsClassifier.pkl")
with open(knn_path, "wb") as f:
    pickle.dump(model_knn, f)
print(f"  → salvato in {knn_path}")

y_pred_knn = model_knn.predict(X_test)
print("\nKNN sklearn")
print(classification_report(y_test, y_pred_knn,
                             target_names=list(cfg.labels.values()), zero_division=0))
print_confusion_matrix(y_test, y_pred_knn, "KNN sklearn")

# ---------------------------------------------------------------------------
# Preparazione dati per deep learning
# ---------------------------------------------------------------------------

X_train_deep = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_deep  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))
y_train_cat  = to_categorical(y_train, num_classes=n_classes)
y_test_cat   = to_categorical(y_test,  num_classes=n_classes)

# ---------------------------------------------------------------------------
# 4c. CNN (3-branch multi-scale)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training CNN …")
print("=" * 60)

def _cnn_branch(inputs, kernel_size):
    x = keras.layers.Conv1D(64, kernel_size=kernel_size, activation='relu')(inputs)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    return keras.layers.Flatten()(x)

inp = [keras.layers.Input(X_train_deep.shape[1:]) for _ in range(3)]
branches = [_cnn_branch(inp[i], ks) for i, ks in enumerate([3, 5, 11])]
merged   = keras.layers.concatenate(branches)
dense    = keras.layers.Dense(100, activation='relu')(merged)
out_cnn  = keras.layers.Dense(n_classes, activation='softmax')(dense)

cnn_model = keras.Model(inputs=inp, outputs=out_cnn)
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.summary()

history_cnn = cnn_model.fit(
    [X_train_deep] * 3, y_train_cat,
    epochs=cfg.cnn_epochs,
    batch_size=cfg.cnn_batch_size,
    validation_split=0.2,
    verbose=1
)

cnn_path = os.path.join(cfg.models_dir, "CNN.keras")
cnn_model.save(cnn_path)
print(f"  → salvato in {cnn_path}")
plot_training_history(history_cnn, "CNN")

y_pred_cnn  = np.argmax(cnn_model.predict([X_test_deep] * 3), axis=1)
y_test_cls  = np.argmax(y_test_cat, axis=1)
print("\nCNN")
print(classification_report(y_test_cls, y_pred_cnn,
                             target_names=list(cfg.labels.values()), zero_division=0))
print_confusion_matrix(y_test_cls, y_pred_cnn, "CNN")

# ---------------------------------------------------------------------------
# 4d. ResNet 1D
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training ResNet 1D …")
print("=" * 60)

fm = cfg.resnet_n_feature_maps

def _res_block(x_in, filters, expand_channels=False):
    x = keras.layers.Conv1D(filters, 8, padding='same')(x_in)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv1D(filters, 5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv1D(filters, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    sc = (keras.layers.Conv1D(filters, 1, padding='same')(x_in)
          if expand_channels else x_in)
    sc = keras.layers.BatchNormalization()(sc)

    return keras.layers.Activation('relu')(keras.layers.add([sc, x]))

inp_res   = keras.layers.Input(X_train_deep.shape[1:])
b1        = _res_block(inp_res, fm,      expand_channels=True)
b2        = _res_block(b1,      fm * 2,  expand_channels=True)
b3        = _res_block(b2,      fm * 2,  expand_channels=False)
gap       = keras.layers.GlobalAveragePooling1D()(b3)
out_res   = keras.layers.Dense(n_classes, activation='softmax')(gap)

resnet_model = keras.models.Model(inputs=inp_res, outputs=out_res)
resnet_model.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(),
                     metrics=['accuracy'])
resnet_model.summary()

mini_batch = int(min(X_train_deep.shape[0] / cfg.resnet_batch_divisor,
                     cfg.resnet_max_batch))
history_resnet = resnet_model.fit(
    X_train_deep, y_train_cat,
    epochs=cfg.resnet_epochs,
    batch_size=mini_batch,
    validation_split=0.2,
    verbose=1
)

resnet_path = os.path.join(cfg.models_dir, "ResNet.keras")
resnet_model.save(resnet_path)
print(f"  → salvato in {resnet_path}")
plot_training_history(history_resnet, "ResNet")

y_pred_resnet = np.argmax(resnet_model.predict(X_test_deep), axis=1)
print("\nResNet")
print(classification_report(y_test_cls, y_pred_resnet,
                             target_names=list(cfg.labels.values()), zero_division=0))
print_confusion_matrix(y_test_cls, y_pred_resnet, "ResNet")

# ---------------------------------------------------------------------------
# Riepilogo finale
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Training completato. Modelli salvati:")
print(f"  {knn_dtw_path}")
print(f"  {knn_path}")
print(f"  {cnn_path}")
print(f"  {resnet_path}")
print(f"  Confusion matrix e curve → {cfg.models_dir}/")
print("=" * 60)
