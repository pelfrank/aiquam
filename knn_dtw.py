"""
AIQUAM++ — KNN + DTW classifier
Estratto da training.py per riuso in training e inferenza.
"""

import sys
import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import squareform


class ProgressBar:
    """Barra di avanzamento testuale per la matrice DTW."""

    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar   = '[]'
        self.fill_char  = '*'
        self.width      = 40
        self.__update_amount(0)
        self.animate = self.animate_ipython

    def animate_ipython(self, iter):
        print('\r', self, end='', flush=True)
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full     = self.width - 2
        num_hashes   = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place    = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string   = '%d%%' % percent_done
        self.prog_bar = (self.prog_bar[0:pct_place]
                         + pct_string
                         + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


class KnnDtw:
    """
    K-Nearest Neighbors con Dynamic Time Warping come metrica di distanza.

    Parametri
    ----------
    n_neighbors : int
        Numero di vicini da considerare (default 5).
    max_warping_window : int
        Ampiezza della finestra di Sakoe-Chiba per vincolare il percorso DTW
        (default 10000, praticamente illimitato).
    subsample_step : int
        Passo di sottocampionamento delle serie prima del calcolo DTW
        (default 1 = nessun sottocampionamento).
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors        = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step     = subsample_step

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, x, l):
        """Memorizza il training set (lazy learning)."""
        self.x = x
        self.l = l

    # ------------------------------------------------------------------
    # Distanza DTW tra due serie
    # ------------------------------------------------------------------

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])
        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                           min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        return cost[-1, -1]

    # ------------------------------------------------------------------
    # Matrice delle distanze
    # ------------------------------------------------------------------

    def _dist_matrix(self, x, y):
        dm_count = 0

        if np.array_equal(x, y):
            x_s = np.shape(x)
            dm  = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            p   = ProgressBar(dm.shape[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(
                        x[i, ::self.subsample_step],
                        y[j, ::self.subsample_step])
                    dm_count += 1
                    p.animate(dm_count)

            return squareform(dm)
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm  = np.zeros((x_s[0], y_s[0]))
            p   = ProgressBar(x_s[0] * y_s[0])

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(
                        x[i, ::self.subsample_step],
                        y[j, ::self.subsample_step])
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    # ------------------------------------------------------------------
    # Predizione
    # ------------------------------------------------------------------

    def predict(self, x):
        """
        Restituisce (labels, probabilità) per ogni campione in x.
        La probabilità è la frazione di vicini con la label predetta.
        """
        dm = self._dist_matrix(x, self.x)
        knn_idx    = dm.argsort()[:, :self.n_neighbors]
        knn_labels = self.l[knn_idx]
        mode_data  = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors
        return mode_label.ravel(), mode_proba.ravel()
