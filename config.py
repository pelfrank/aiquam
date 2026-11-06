"""
AIQUAM++ — Loader centralizzato della configurazione.

Legge config.ini e ne espone i valori già convertiti nel tipo corretto.
Tutti gli altri script importano da qui invece di leggere config.ini
direttamente.

Uso
---
    from config import cfg_training as cfg
    print(cfg.dataset_dir)          # 'dataset'
    print(cfg.hours)                # 72

    from config import cfg_inference as cfi
    print(cfi.url)                  # 'http://...'
"""

import os
from configparser import ConfigParser

# ---------------------------------------------------------------------------
# Caricamento raw del file .ini
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_INI_PATH = os.path.join(_BASE_DIR, 'config.ini')

_parser = ConfigParser()
_parser.read(_INI_PATH)


def _require(section, key):
    """Legge un valore obbligatorio; solleva un errore chiaro se mancante."""
    if not _parser.has_option(section, key):
        raise KeyError(f"[config.ini] chiave '{key}' mancante nella sezione '[{section}]'")
    return _parser.get(section, key)


# ---------------------------------------------------------------------------
# Configurazione Training
# ---------------------------------------------------------------------------

class _TrainingConfig:
    """Parametri per training_new.py, letti dalla sezione [training]."""

    # Dataset
    dataset_dir   = _require('training', 'DATASET_DIR')
    dataset_years = [y.strip() for y in _require('training', 'DATASET_YEARS').split(',')]
    hours         = _parser.getint('training', 'HOURS')

    # Feature columns: h_-71, h_-70, ..., h_-01, h_+00
    feature_cols = [f"h_-{str(i).zfill(2)}" for i in range(hours - 1, 0, -1)] + ["h_+00"]

    # Classi
    n_classes = _parser.getint('training', 'N_CLASSES')
    labels    = {
        int(pair.split(':')[0].strip()): pair.split(':')[1].strip()
        for pair in _require('training', 'CLASS_LABELS').split(',')
    }

    # Output
    models_dir = _require('training', 'MODELS_DIR')

    # Riproducibilità
    random_seed = _parser.getint('training', 'RANDOM_SEED')
    test_size   = _parser.getfloat('training', 'TEST_SIZE')

    # KNN + DTW
    knn_dtw_n_neighbors    = _parser.getint('training', 'KNN_DTW_N_NEIGHBORS')
    knn_dtw_warping_window = _parser.getint('training', 'KNN_DTW_WARPING_WINDOW')

    # KNN sklearn
    knn_n_neighbors = _parser.getint('training', 'KNN_N_NEIGHBORS')

    # CNN
    cnn_epochs     = _parser.getint('training', 'CNN_EPOCHS')
    cnn_batch_size = _parser.getint('training', 'CNN_BATCH_SIZE')

    # ResNet
    resnet_epochs         = _parser.getint('training', 'RESNET_EPOCHS')
    resnet_n_feature_maps = _parser.getint('training', 'RESNET_N_FEATURE_MAPS')
    resnet_batch_divisor  = _parser.getint('training', 'RESNET_BATCH_DIVISOR')
    resnet_max_batch      = _parser.getint('training', 'RESNET_MAX_BATCH')


# ---------------------------------------------------------------------------
# Configurazione Inferenza
# ---------------------------------------------------------------------------

class _InferenceConfig:
    """Parametri per prediction.py, letti dalla sezione [inference]."""

    url         = _require('inference', 'URL')
    sample_hour = _require('inference', 'SAMPLE_HOUR')
    start       = _parser.getint('inference', 'START')
    end         = _parser.getint('inference', 'END')
    areas_file  = _require('inference', 'AREAS_FILE')
    output_dir  = _require('inference', 'OUTPUT_DIR')
    # Directory locale con NetCDF pre-scaricati (vuota = usa sempre OPeNDAP)
    nc_root = _parser.get('inference', 'NC_ROOT').strip() or None


# ---------------------------------------------------------------------------
# Istanze pubbliche — da importare negli altri script
# ---------------------------------------------------------------------------

cfg_training  = _TrainingConfig()
cfg_inference = _InferenceConfig()


# ---------------------------------------------------------------------------
# Esecuzione diretta: stampa il riepilogo della configurazione attiva
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 55)
    print("AIQUAM++ — Configurazione attiva")
    print(f"  File letto: {_INI_PATH}")
    print("=" * 55)

    print("\n[training]")
    t = cfg_training
    print(f"  dataset_dir    = {t.dataset_dir}")
    print(f"  dataset_years  = {t.dataset_years}")
    print(f"  hours          = {t.hours}")
    print(f"  feature_cols   = {t.feature_cols[0]} … {t.feature_cols[-1]}  ({len(t.feature_cols)} col)")
    print(f"  n_classes      = {t.n_classes}")
    print(f"  labels         = {t.labels}")
    print(f"  models_dir     = {t.models_dir}")
    print(f"  random_seed    = {t.random_seed}")
    print(f"  test_size      = {t.test_size}")
    print(f"  KNN+DTW        k={t.knn_dtw_n_neighbors}, window={t.knn_dtw_warping_window}")
    print(f"  KNN            k={t.knn_n_neighbors}")
    print(f"  CNN            epochs={t.cnn_epochs}, batch={t.cnn_batch_size}")
    print(f"  ResNet         epochs={t.resnet_epochs}, maps={t.resnet_n_feature_maps}, "
          f"batch=min(n/{t.resnet_batch_divisor}, {t.resnet_max_batch})")

    print("\n[inference]")
    i = cfg_inference
    print(f"  url            = {i.url}")
    print(f"  start/end      = {i.start} … {i.end}")
    print(f"  areas_file     = {i.areas_file}")
    print(f"  output_dir     = {i.output_dir}")
    print(f"  nc_root        = {i.nc_root or '(non impostato — usa OPeNDAP)'}")
    print()
