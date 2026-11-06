"""
AIQUAM++ — Predizione da campione CSV

Modalità singolo file:
    python predict_csv.py file <input.csv> [output.csv]
    python predict_csv.py <input.csv> [output.csv]       # retrocompatibile

    Legge un CSV nel formato del training (colonne h_-71 … h_+00, opzionale 'target')
    e restituisce un CSV separato con tutte le colonne originali più le predizioni
    dei 4 modelli e il voto di maggioranza (predicted_class).

    Se output.csv non viene specificato, viene creato accanto all'input con suffisso
    '_predicted.csv'.

Modalità directory:
    python predict_csv.py dir <input_dir> [output_dir]

    Elabora tutti i file *.csv presenti in <input_dir> (ricorsivamente se
    --recursive è specificato) e salva i risultati in <output_dir>.
    Se output_dir è omesso, i file vengono salvati nella stessa directory
    dell'input con suffisso '_predicted.csv'.

    Opzioni:
        --recursive   cerca i CSV anche nelle sottodirectory
        --skip-errors continua in caso di errore su un singolo file
"""

import os
import sys
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
from tensorflow import keras

from config import cfg_training as cfg


# ---------------------------------------------------------------------------
# Caricamento modelli (una sola volta, condiviso tra tutte le modalità)
# ---------------------------------------------------------------------------

def load_models():
    models_dir = cfg.models_dir
    knn_dtw_path = os.path.join(models_dir, "KNN_DTW.pkl")
    knn_path     = os.path.join(models_dir, "KNeighborsClassifier.pkl")
    cnn_path     = os.path.join(models_dir, "CNN.keras")
    resnet_path  = os.path.join(models_dir, "ResNet.keras")

    for p in [knn_dtw_path, knn_path, cnn_path, resnet_path]:
        if not os.path.exists(p):
            print(f"  Errore: modello non trovato: {p}")
            sys.exit(1)

    print("Caricamento modelli...")
    model_knn_dtw = pickle.load(open(knn_dtw_path, "rb"))
    model_knn     = pickle.load(open(knn_path, "rb"))
    model_cnn     = keras.models.load_model(cnn_path)
    model_resnet  = keras.models.load_model(resnet_path)
    print("  Modelli caricati.\n")

    return model_knn_dtw, model_knn, model_cnn, model_resnet


# ---------------------------------------------------------------------------
# Predizione su un DataFrame già caricato
# ---------------------------------------------------------------------------

def predict_df(df, models):
    model_knn_dtw, model_knn, model_cnn, model_resnet = models

    missing = [c for c in cfg.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    X_raw = df[cfg.feature_cols].values.astype(np.float32)
    X     = np.where(np.isfinite(X_raw), X_raw, 0.0)

    # KNN + DTW
    y_knn_dtw, _ = model_knn_dtw.predict(X)
    y_knn_dtw    = np.array(y_knn_dtw, dtype=int)

    # KNN sklearn
    y_knn = model_knn.predict(X).astype(int)

    # CNN (3 branch)
    X_deep = X.reshape((X.shape[0], X.shape[1], 1))
    y_cnn  = np.argmax(model_cnn.predict([X_deep] * 3, verbose=0), axis=1).astype(int)

    # ResNet
    y_resnet = np.argmax(model_resnet.predict(X_deep, verbose=0), axis=1).astype(int)

    # Voto di maggioranza tra i 4 modelli
    votes   = np.stack([y_knn_dtw, y_knn, y_cnn, y_resnet], axis=1)
    y_final = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=cfg.n_classes).argmax(),
        axis=1,
        arr=votes,
    ).astype(int)

    df_out = df.copy()
    df_out["pred_knn_dtw"]    = y_knn_dtw
    df_out["pred_knn"]        = y_knn
    df_out["pred_cnn"]        = y_cnn
    df_out["pred_resnet"]     = y_resnet
    df_out["predicted_class"] = y_final

    return df_out, y_final


def print_distribution(y_final):
    print("  Distribuzione predicted_class:")
    for cls, label in cfg.labels.items():
        n = (y_final == cls).sum()
        print(f"    Classe {cls} ({label}): {n} campioni")


def output_path_for(input_path, output_dir=None):
    base, ext = os.path.splitext(input_path)
    suffix = "_predicted" + (ext if ext else ".csv")
    if output_dir:
        fname = os.path.basename(base) + suffix
        return os.path.join(output_dir, fname)
    return base + suffix


# ---------------------------------------------------------------------------
# Modalità singolo file
# ---------------------------------------------------------------------------

def run_file(input_path, output_path, models):
    print(f"Caricamento: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  {len(df)} righe  |  {len(cfg.feature_cols)} feature per riga")

    print("  Predizione...")
    df_out, y_final = predict_df(df, models)

    df_out.to_csv(output_path, index=False)
    print(f"  Salvato: {output_path}")
    print_distribution(y_final)


# ---------------------------------------------------------------------------
# Modalità directory
# ---------------------------------------------------------------------------

def run_dir(input_dir, output_dir, recursive, skip_errors, models):
    if not os.path.isdir(input_dir):
        print(f"Errore: '{input_dir}' non è una directory.")
        sys.exit(1)

    pattern = os.path.join(input_dir, "**", "*.csv") if recursive \
              else os.path.join(input_dir, "*.csv")
    csv_files = sorted(glob.glob(pattern, recursive=recursive))

    # Esclude eventuali file già predetti (con suffisso _predicted)
    csv_files = [f for f in csv_files if not os.path.basename(f).endswith("_predicted.csv")]

    if not csv_files:
        print(f"Nessun file CSV trovato in '{input_dir}'.")
        sys.exit(0)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Trovati {len(csv_files)} file CSV in '{input_dir}'.\n")

    ok, errors = 0, []
    for i, csv_path in enumerate(csv_files, 1):
        rel = os.path.relpath(csv_path, input_dir)
        out = output_path_for(csv_path, output_dir)

        # Se l'output è in una sottodirectory diversa, creala
        os.makedirs(os.path.dirname(out), exist_ok=True)

        print(f"[{i}/{len(csv_files)}] {rel}")
        try:
            df = pd.read_csv(csv_path)
            print(f"  {len(df)} righe")
            df_out, y_final = predict_df(df, models)
            df_out.to_csv(out, index=False)
            print(f"  → {out}")
            print_distribution(y_final)
            ok += 1
        except Exception as e:
            msg = f"  ERRORE su {rel}: {e}"
            print(msg)
            errors.append((rel, str(e)))
            if not skip_errors:
                print("  Interrotto. Usa --skip-errors per continuare sugli altri file.")
                sys.exit(1)
        print()

    print(f"Completato: {ok}/{len(csv_files)} file elaborati con successo.")
    if errors:
        print(f"File con errori ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Costruisce il parser con due sottocomandi: "file" e "dir"
    # Mantiene la retrocompatibilità: se il primo argomento non è "file"/"dir"
    # lo tratta come un percorso CSV (vecchio comportamento).
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Retrocompatibilità: primo argomento non è un sottocomando noto
    if sys.argv[1] not in ("file", "dir"):
        # Comportamento originale: predict_csv.py <input.csv> [output.csv]
        input_path  = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) >= 3 \
                      else output_path_for(input_path)
        models = load_models()
        run_file(input_path, output_path, models)
        return

    parser = argparse.ArgumentParser(
        prog="predict_csv.py",
        description="AIQUAM++ — predizione su CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- sottocomando file ---
    p_file = sub.add_parser("file", help="Predizione su un singolo CSV")
    p_file.add_argument("input",  help="File CSV di input")
    p_file.add_argument("output", nargs="?", default=None, help="File CSV di output (opzionale)")

    # --- sottocomando dir ---
    p_dir = sub.add_parser("dir", help="Predizione su tutti i CSV di una directory")
    p_dir.add_argument("input_dir",  help="Directory di input")
    p_dir.add_argument("output_dir", nargs="?", default=None,
                       help="Directory di output (opzionale; default: stessa dell'input)")
    p_dir.add_argument("--recursive",   action="store_true",
                       help="Cerca CSV anche nelle sottodirectory")
    p_dir.add_argument("--skip-errors", action="store_true",
                       help="Continua in caso di errore su un singolo file")

    args = parser.parse_args()

    models = load_models()

    if args.cmd == "file":
        out = args.output or output_path_for(args.input)
        run_file(args.input, out, models)

    elif args.cmd == "dir":
        run_dir(
            args.input_dir,
            args.output_dir,
            recursive=args.recursive,
            skip_errors=args.skip_errors,
            models=models,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
