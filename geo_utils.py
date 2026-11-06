"""
AIQUAM++ — Utility geografiche per l'inferenza.
Estratto da main.py: solo le funzioni usate da prediction.py.
"""

import json
import math
import numpy as np
import netCDF4

from config import cfg_inference as cfg


def load_areas():
    """Carica le aree di allevamento dal file GeoJSON."""
    with open(cfg.areas_file) as f:
        data = json.load(f)
    return data["features"]


def get_index_lat_long(lat, long, delta_lat, delta_long,
                       min_lat, max_lat, min_long, max_long):
    """
    Converte coordinate geografiche in indici nella griglia del modello WCM3.
    Restituisce (index_min_lat, index_max_lat, index_min_long, index_max_long).
    """
    index_min_lat  = round((min_lat  - lat[0])  / delta_lat)
    index_max_lat  = round((max_lat  - lat[0])  / delta_lat)
    index_min_long = round((min_long - long[0]) / delta_long)
    index_max_long = round((max_long - long[0]) / delta_long)

    return index_min_lat, index_max_lat, index_min_long, index_max_long


def get_lat_long():
    """
    Legge latitudine, longitudine e passi della griglia da un file NetCDF
    di riferimento sul server OPeNDAP.
    Restituisce (lat, long, delta_lat, delta_long) come array numpy.
    """
    url = cfg.url + "2020/01/01/wcm3_d03_20200101Z0100.nc"
    _dataset = netCDF4.Dataset(url)

    lat  = _dataset.variables['latitude'][:]
    long = _dataset.variables['longitude'][:]

    N = len(lat)
    delta_lat = sum(lat[i] - lat[i - 1] for i in range(1, N)) / N

    M = len(long)
    delta_long = sum(long[i] - long[i - 1] for i in range(1, M)) / M

    return np.array(lat), np.array(long), delta_lat, delta_long
