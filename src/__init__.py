"""
Sistema Inteligente de Vigilancia y Asistencia para la Calidad del Aire en el Valle de Aburrá.
Proyecto Final Integrador SI3003 - Inteligencia Artificial | Universidad EAFIT.
"""

import os

# Prevención de colisión de librerías OpenMP (PyTorch + FAISS) en macOS Apple Silicon y Linux
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "1.0.0"
