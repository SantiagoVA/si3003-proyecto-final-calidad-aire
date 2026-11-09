"""
Parte 1 (Fundamentos Clásicos de IA): Modelos Probabilísticos Implementados Desde Cero en NumPy.

Incluye implementaciones 100% propias (sin invocar `.fit()` de scikit-learn ni librerías de alto nivel):
1. `MajorityClassBaseline`: Línea base de clase mayoritaria empírica $P(Y=k)$.
2. `PersistenceBaseline`: Línea base física de inercia atmosférica ($\hat{y}_{t+1} = y_t$).
3. `GaussianNaiveBayesScratch`: Clasificador Bayesiano Ingenuo Gaussiano en espacio logarítmico
   con suavizado de Dirichlet/Laplace en prioris y estabilización numérica `log-sum-exp`.
4. `GaussianHMMScratch`: Modelo Oculto de Markov con emisiones Gaussianas multivariadas,
   estimación de matriz estocástica de transición $A_{ij} = P(S_{t+1}=j \mid S_t=i)$,
   **Filtrado Recursivo Forward ($\alpha_t$)**, **Decodificación de Viterbi (Programación Dinámica)**
   y **Predicción Bayesiana Dinámica a un paso ($t+1$)** acoplando inercia de Markov y meteorología.
5. Métricas cuantitativas implementadas desde cero en NumPy: Accuracy, Matriz de Confusión,
   Precision, Recall, Macro-F1, Weighted-F1 y Brier Score multiclase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import (
    CATEGORY_TO_COLOR,
    PROCESSED_DIR,
    STATIONS_METADATA,
    load_air_quality_data,
    resolve_station_name,
)

MODEL_ARTIFACTS_JSON = PROCESSED_DIR / "classical_model_params.json"

# Clases ordenadas por severidad ambiental según Resolución 2254 de 2017 / POECA
STATE_LABELS: List[str] = [
    "Buena",                     # 0 - Verde (0 - 12.0 ug/m3)
    "Aceptable",                 # 1 - Amarillo (12.1 - 37.0 ug/m3)
    "Dañina a Grupos Sensibles", # 2 - Naranja (37.1 - 55.4 ug/m3) -> Prevención POECA
    "Dañina a la Salud",         # 3 - Rojo (>= 55.5 ug/m3) -> Alerta POECA
]

FEATURE_COLUMNS: List[str] = [
    "pm25",                # Concentración actual 24h (ug/m3)
    "pm25_previo",         # Concentración rezagada t-1 (ug/m3)
    "pm25_ma3",            # Media móvil últimos 3 días (ug/m3)
    "delta_pm25",          # Tendencia diaria PM2.5_t - PM2.5_{t-1}
    "pm25_base_estacion",  # Línea base histórica de la estación en entrenamiento (ug/m3)
    "anomalia_estacion",   # Anomalía local PM2.5_t - base_estacion (ug/m3)
    "velocidad_viento",    # Velocidad media del viento (m/s)
    "humedad_relativa",    # Humedad relativa media (%)
    "temperatura",         # Temperatura superficial media (°C)
    "precipitacion",       # Precipitación acumulada diaria (mm)
    "es_periodo_poeca",    # Indicador bimodal feb-abr / oct-nov (0 o 1)
]


def _logsumexp_np(log_probs: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Implementación numéricamente estable de log(sum(exp(x))) en NumPy puro."""
    max_val = np.max(log_probs, axis=axis, keepdims=True)
    stabilized = log_probs - max_val
    res = max_val + np.log(np.sum(np.exp(stabilized), axis=axis, keepdims=True))
    if not keepdims:
        res = np.squeeze(res, axis=axis)
    return res


def encode_labels(y_str: pd.Series | np.ndarray) -> np.ndarray:
    """Mapea nombres de categoría ICA a índices enteros {0, 1, 2, 3}."""
    mapping = {
        "Buena": 0,
        "Aceptable": 1,
        "Dañina a Grupos Sensibles": 2,
        "Dañina a la Salud": 3,
        "Muy Dañina a la Salud": 3,
        "Peligrosa": 3,
    }
    return np.array([mapping.get(str(v), 1) for v in y_str], dtype=int)


def decode_labels(y_idx: np.ndarray) -> List[str]:
    return [STATE_LABELS[int(i)] for i in y_idx]


def prepare_features_and_targets(
    df: pd.DataFrame,
    split_date: str = "2025-01-01",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Construye la matriz de características $X_t$, el estado actual $y_t$ y el estado objetivo
    del día siguiente $y_{t+1}$ respetando estrictamente la causalidad temporal por estación.
    La media base por estación se estima únicamente con los datos anteriores a `split_date`.
    """
    work = df.sort_values(["estacion", "fecha"]).copy()
    work["es_periodo_poeca"] = work["mes"].isin([2, 3, 4, 10, 11]).astype(float)
    work = work.dropna(subset=["categoria_manana"]).reset_index(drop=True)

    train_sub = work[work["fecha"] < split_date]
    if train_sub.empty:
        train_sub = work
    station_base = train_sub.groupby("estacion")["pm25"].mean().to_dict()
    global_base = float(train_sub["pm25"].mean())

    work["pm25_base_estacion"] = work["estacion"].map(station_base).fillna(global_base)
    work["anomalia_estacion"] = work["pm25"] - work["pm25_base_estacion"]

    X = work[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_today = encode_labels(work["categoria_ica"].to_numpy())
    y_tomorrow = encode_labels(work["categoria_manana"].to_numpy())
    return work, X, y_today, y_tomorrow, station_base


def compute_classification_metrics_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_classes: int = 4,
) -> Dict:
    """
    Calcula métricas de clasificación multiclase 100% desde cero en NumPy:
    Accuracy, Matriz de Confusión, Precision, Recall, F1 por clase, Macro-F1, Weighted-F1 y Brier Score.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    accuracy = float(np.trace(cm) / max(n, 1))

    per_class = {}
    f1_present = []
    f1_all = []
    supports = np.sum(cm, axis=1)

    for k in range(n_classes):
        tp = float(cm[k, k])
        fp = float(np.sum(cm[:, k]) - tp)
        fn = float(np.sum(cm[k, :]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_all.append(f1)
        if supports[k] > 0:
            f1_present.append(f1)
        per_class[STATE_LABELS[k]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(supports[k]),
        }

    macro_f1 = float(np.mean(f1_present)) if f1_present else 0.0
    weighted_f1 = float(np.sum(np.array(f1_all) * supports) / max(np.sum(supports), 1))

    brier = None
    if y_proba is not None:
        one_hot = np.eye(n_classes)[y_true]
        brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "brier_score": round(brier, 4) if brier is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


class MajorityClassBaseline:
    """Línea Base 1: Predice siempre la categoría más frecuente del conjunto de entrenamiento."""

    def __init__(self, n_classes: int = 4):
        self.n_classes = n_classes
        self.class_priors_: np.ndarray = np.ones(n_classes) / n_classes
        self.majority_class_: int = 1

    def fit(self, y_train: np.ndarray) -> "MajorityClassBaseline":
        counts = np.bincount(y_train, minlength=self.n_classes).astype(float)
        self.class_priors_ = counts / np.sum(counts)
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict_proba(self, n_samples: int) -> np.ndarray:
        return np.tile(self.class_priors_, (n_samples, 1))

    def predict(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.majority_class_, dtype=int)


class PersistenceBaseline:
    """
    Línea Base 2 (Física / Meteorológica): Modelo de Persistencia ($\hat{y}_{t+1} = y_t$).
    Asume que la categoría ICA de mañana será idéntica a la categoría ICA observada hoy.
    """

    def __init__(self, n_classes: int = 4, smoothing: float = 0.05):
        self.n_classes = n_classes
        self.smoothing = smoothing

    def predict_proba(self, y_today: np.ndarray) -> np.ndarray:
        probs = np.full((len(y_today), self.n_classes), self.smoothing / (self.n_classes - 1))
        for i, state in enumerate(y_today):
            probs[i, int(state)] = 1.0 - self.smoothing
        return probs

    def predict(self, y_today: np.ndarray) -> np.ndarray:
        return np.asarray(y_today, dtype=int).copy()


class GaussianNaiveBayesScratch:
    """
    Clasificador Bayesiano Ingenuo Gaussiano implementado 100% desde cero en NumPy.

    Formulación matemática:
        P(Y = k | x) \\propto P(Y = k)^\\gamma \\prod_{j=1}^d \\mathcal{N}(x_j \\mid \\mu_{kj}, \\sigma^2_{kj} + \\epsilon)^\\tau
    En espacio logarítmico:
        \\log P(Y = k | x) = \\gamma \\log \\pi_k - \\frac{\\tau}{2} \\sum_{j=1}^d \\left[ \\log(2\\pi \\tilde{\\sigma}^2_{kj}) + \\frac{(x_j - \\mu_{kj})^2}{\\tilde{\\sigma}^2_{kj}} \\right] - Z(x)
    donde $\\tau \\in (0, 1]$ atenúa la colinealidad entre rezagos de $PM_{2.5}$ y variables climáticas.
    """

    def __init__(
        self,
        n_classes: int = 4,
        var_smoothing: float = 1e-2,
        prior_alpha: float = 1.0,
        likelihood_temp: float = 0.50,
        prior_weight: float = 0.80,
    ):
        self.n_classes = n_classes
        self.var_smoothing = var_smoothing
        self.prior_alpha = prior_alpha
        self.likelihood_temp = likelihood_temp
        self.prior_weight = prior_weight
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.vars_: Optional[np.ndarray] = None
        self.feat_mean_: Optional[np.ndarray] = None
        self.feat_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesScratch":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        # Estandarización z-score propia calculada exclusivamente en entrenamiento
        self.feat_mean_ = np.mean(X, axis=0)
        self.feat_std_ = np.std(X, axis=0)
        self.feat_std_[self.feat_std_ < 1e-8] = 1.0
        Xz = (X - self.feat_mean_) / self.feat_std_

        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        self.priors_ = (counts + self.prior_alpha) / (n_samples + self.n_classes * self.prior_alpha)

        self.means_ = np.zeros((self.n_classes, n_features), dtype=float)
        self.vars_ = np.ones((self.n_classes, n_features), dtype=float)

        global_mean = np.mean(Xz, axis=0)
        global_var = np.var(Xz, axis=0) + self.var_smoothing

        for k in range(self.n_classes):
            mask = y == k
            nk = int(np.sum(mask))
            if nk >= 2:
                # Contracción empírica suave hacia la media global si una clase extrema tiene pocos casos
                shrink = nk / (nk + 3.0)
                self.means_[k] = shrink * np.mean(Xz[mask], axis=0) + (1.0 - shrink) * global_mean
                self.vars_[k] = shrink * np.var(Xz[mask], axis=0) + (1.0 - shrink) * global_var + self.var_smoothing
            elif nk == 1:
                self.means_[k] = 0.5 * Xz[mask][0] + 0.5 * global_mean
                self.vars_[k] = global_var
            else:
                self.means_[k] = global_mean
                self.vars_[k] = global_var

        return self

    def _log_gaussian_likelihood(self, X: np.ndarray) -> np.ndarray:
        Xz = (np.asarray(X, dtype=float) - self.feat_mean_) / self.feat_std_
        # Xz: (N, D), means_: (K, D), vars_: (K, D) -> diff: (N, K, D)
        diff = Xz[:, None, :] - self.means_[None, :, :]
        log_det = np.sum(np.log(2.0 * np.pi * self.vars_), axis=1)  # (K,)
        mahal = np.sum((diff ** 2) / self.vars_[None, :, :], axis=2)  # (N, K)
        return -0.5 * (log_det[None, :] + mahal)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_lik = self._log_gaussian_likelihood(X)
        log_prior = np.log(np.clip(self.priors_, 1e-12, 1.0))
        joint_log = self.prior_weight * log_prior[None, :] + self.likelihood_temp * log_lik
        log_norm = _logsumexp_np(joint_log, axis=1, keepdims=True)
        return np.exp(joint_log - log_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class GaussianHMMScratch:
    """
    Modelo Oculto de Markov con Emisiones Gaussianas y Predictor Bayesiano Dinámico
    implementado 100% desde cero en NumPy.

    Componentes matemáticos:
    1. Distribución inicial de estados: $\\pi_k = P(S_1 = k)$.
    2. Matriz estocástica de transición de Markov entre días consecutivos:
       $A_{ij} = P(S_{t+1} = j \\mid S_t = i)$ con suavizado de Dirichlet $\\alpha_A$.
    3. Verosimilitud de emisión Gaussiana $B_k(x_t) = p(x_t \\mid S_t = k)$ y modelo
       predictivo condicional meteorológico $L_k(x_t) = p(x_t \\mid S_{t+1} = k)$.
    4. Algoritmo Forward ($\\alpha_t$): filtrado recursivo de estado latente por estación:
       $\\alpha_t(j) \\propto B_j(x_t) \\sum_{i=1}^K \\alpha_{t-1}(i) A_{ij}$.
    5. Algoritmo de Viterbi: programación dinámica para hallar la trayectoria más probable
       $s^*_{1:T} = \\arg\\max_{s_{1:T}} P(S_{1:T} = s_{1:T} \\mid x_{1:T})$.
    6. Distribución predictiva a un paso ($t+1$):
       $P(S_{t+1} = k \\mid x_{1:t}) \\propto \\left[\\sum_{i=1}^K \\alpha_t(i) A_{ik}\\right]^{\\omega} \\cdot L_k(x_t)^{\\eta}$
    """

    def __init__(
        self,
        n_states: int = 4,
        trans_alpha: float = 0.5,
        var_smoothing: float = 2e-2,
        markov_weight: float = 1.0,
        meteo_likelihood_temp: float = 0.25,
    ):
        self.n_states = n_states
        self.trans_alpha = trans_alpha
        self.var_smoothing = var_smoothing
        self.markov_weight = markov_weight
        self.meteo_likelihood_temp = meteo_likelihood_temp

        self.startprob_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.emission_means_: Optional[np.ndarray] = None
        self.emission_vars_: Optional[np.ndarray] = None
        self.next_cond_means_: Optional[np.ndarray] = None
        self.next_cond_vars_: Optional[np.ndarray] = None
        self.feat_mean_: Optional[np.ndarray] = None
        self.feat_std_: Optional[np.ndarray] = None
        # Pesos discriminativos aprendidos por gradiente descendente L2 en NumPy (acoplamiento log-lineal)
        self.W_disc_: Optional[np.ndarray] = None
        self.b_disc_: Optional[np.ndarray] = None
        self.disc_blend_: float = 0.65

    def fit(
        self,
        X: np.ndarray,
        y_today: np.ndarray,
        y_tomorrow: np.ndarray,
        station_groups: Optional[np.ndarray] = None,
    ) -> "GaussianHMMScratch":
        X = np.asarray(X, dtype=float)
        y_today = np.asarray(y_today, dtype=int)
        y_tomorrow = np.asarray(y_tomorrow, dtype=int)
        n_samples, n_features = X.shape

        self.feat_mean_ = np.mean(X, axis=0)
        self.feat_std_ = np.std(X, axis=0)
        self.feat_std_[self.feat_std_ < 1e-8] = 1.0
        Xz = (X - self.feat_mean_) / self.feat_std_

        # 1. Distribución estacionaria inicial \pi
        init_counts = np.bincount(y_today, minlength=self.n_states).astype(float) + 0.5
        self.startprob_ = init_counts / np.sum(init_counts)

        # 2. Matriz de transición de Markov A_{ij} = P(S_{t+1} = j | S_t = i)
        trans_counts = np.full((self.n_states, self.n_states), self.trans_alpha, dtype=float)
        for s_t, s_next in zip(y_today, y_tomorrow):
            trans_counts[int(s_t), int(s_next)] += 1.0
        self.transmat_ = trans_counts / np.sum(trans_counts, axis=1, keepdims=True)

        # 3. Emisiones Gaussianas p(x_t | S_t = k) y verosimilitud predictiva p(x_t | S_{t+1} = k)
        self.emission_means_ = np.zeros((self.n_states, n_features), dtype=float)
        self.emission_vars_ = np.ones((self.n_states, n_features), dtype=float)
        self.next_cond_means_ = np.zeros((self.n_states, n_features), dtype=float)
        self.next_cond_vars_ = np.ones((self.n_states, n_features), dtype=float)

        g_mean = np.mean(Xz, axis=0)
        g_var = np.var(Xz, axis=0) + self.var_smoothing

        for k in range(self.n_states):
            m_cur = y_today == k
            nk_cur = int(np.sum(m_cur))
            if nk_cur >= 2:
                sh = nk_cur / (nk_cur + 3.0)
                self.emission_means_[k] = sh * np.mean(Xz[m_cur], axis=0) + (1.0 - sh) * g_mean
                self.emission_vars_[k] = sh * np.var(Xz[m_cur], axis=0) + (1.0 - sh) * g_var + self.var_smoothing
            else:
                self.emission_means_[k] = g_mean
                self.emission_vars_[k] = g_var

            m_nxt = y_tomorrow == k
            nk_nxt = int(np.sum(m_nxt))
            if nk_nxt >= 2:
                sh2 = nk_nxt / (nk_nxt + 3.0)
                self.next_cond_means_[k] = sh2 * np.mean(Xz[m_nxt], axis=0) + (1.0 - sh2) * g_mean
                self.next_cond_vars_[k] = sh2 * np.var(Xz[m_nxt], axis=0) + (1.0 - sh2) * g_var + self.var_smoothing
            else:
                self.next_cond_means_[k] = g_mean
                self.next_cond_vars_[k] = g_var

        # 4. Calibración log-lineal discriminativa con pesos balanceados por clase (NumPy puro)
        # Combina one-hot del estado actual S_t + vector meteorológico estandarizado Xz
        one_hot_today = np.eye(self.n_states)[y_today]
        Phi = np.hstack([Xz, one_hot_today])  # (N, D + K)
        dim_phi = Phi.shape[1]

        W = np.zeros((dim_phi, self.n_states), dtype=float)
        b = np.log(self.startprob_ + 1e-8)

        # Ponderación moderada por severidad para preservar sensibilidad en episodios POECA sin sobre-alarmar
        cls_weights = np.array([1.05, 0.95, 1.35, 1.25], dtype=float)[: self.n_states]
        sample_w = cls_weights[y_tomorrow]  # (N,)
        Y_oh = np.eye(self.n_states)[y_tomorrow]

        lr = 0.15
        l2_reg = 2.0e-3
        for epoch in range(450):
            logits = Phi @ W + b[None, :]
            log_z = _logsumexp_np(logits, axis=1, keepdims=True)
            probs = np.exp(logits - log_z)
            err = (probs - Y_oh) * sample_w[:, None] / n_samples
            grad_W = Phi.T @ err + l2_reg * W
            grad_b = np.sum(err, axis=0)
            step = lr / (1.0 + 0.003 * epoch)
            W -= step * grad_W
            b -= step * grad_b

        self.W_disc_ = W
        self.b_disc_ = b
        return self

    def _log_diag_gaussian(self, Xz: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        diff = Xz[:, None, :] - means[None, :, :]
        log_det = np.sum(np.log(2.0 * np.pi * vars_), axis=1)
        mahal = np.sum((diff ** 2) / vars_[None, :, :], axis=2)
        return -0.5 * (log_det[None, :] + mahal)

    def forward_filter_sequence(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Algoritmo Forward recursivo en espacio logarítmico sobre una secuencia de una estación:
            \\alpha_t(j) = P(S_t = j \\mid x_{1:t})
        """
        Xz = (np.asarray(X_seq, dtype=float) - self.feat_mean_) / self.feat_std_
        log_B = self.meteo_likelihood_temp * self._log_diag_gaussian(Xz, self.emission_means_, self.emission_vars_)
        log_A = np.log(np.clip(self.transmat_, 1e-12, 1.0))
        log_pi = np.log(np.clip(self.startprob_, 1e-12, 1.0))

        T = len(Xz)
        alphas = np.zeros((T, self.n_states), dtype=float)

        log_alpha = log_pi + log_B[0]
        log_alpha -= _logsumexp_np(log_alpha)
        alphas[0] = np.exp(log_alpha)

        for t in range(1, T):
            # \sum_i \alpha_{t-1}(i) A_{ij} en log-espacio: logsumexp_i (log_alpha[i] + log_A[i, j])
            log_pred = _logsumexp_np(log_alpha[:, None] + log_A, axis=0)
            log_alpha = log_pred + log_B[t]
            log_alpha -= _logsumexp_np(log_alpha)
            alphas[t] = np.exp(log_alpha)

        return alphas

    def viterbi_decode(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Algoritmo de Viterbi (Programación Dinámica) desde cero en NumPy:
            \\delta_t(j) = \\max_i [\\delta_{t-1}(i) + \\log A_{ij}] + \\log B_j(x_t)
        """
        Xz = (np.asarray(X_seq, dtype=float) - self.feat_mean_) / self.feat_std_
        log_B = self.meteo_likelihood_temp * self._log_diag_gaussian(Xz, self.emission_means_, self.emission_vars_)
        log_A = np.log(np.clip(self.transmat_, 1e-12, 1.0))
        log_pi = np.log(np.clip(self.startprob_, 1e-12, 1.0))

        T = len(Xz)
        delta = np.zeros((T, self.n_states), dtype=float)
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = log_pi + log_B[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_A  # (K, K)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_B[t]

        states = np.zeros(T, dtype=int)
        states[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_next_day_proba(
        self,
        X: np.ndarray,
        y_today: Optional[np.ndarray] = None,
        station_groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calcula la distribución de probabilidad posterior del estado ICA del día siguiente $P(S_{t+1} \\mid x_{1:t})$.
        Integra el prior dinámico de Markov con la verosimilitud Gaussiana meteorológica y calibración log-lineal.
        """
        X = np.asarray(X, dtype=float)
        Xz = (X - self.feat_mean_) / self.feat_std_
        N = len(Xz)

        # Creencia sobre el estado actual \alpha_t
        if station_groups is not None:
            belief_today = np.zeros((N, self.n_states), dtype=float)
            for st in np.unique(station_groups):
                idx = np.where(station_groups == st)[0]
                belief_today[idx] = self.forward_filter_sequence(X[idx])
        elif y_today is not None:
            belief_today = 0.85 * np.eye(self.n_states)[y_today] + 0.15 * self.forward_filter_sequence(X)
        else:
            # Inferencia puntual a partir de la emisión actual
            log_em = self.meteo_likelihood_temp * self._log_diag_gaussian(Xz, self.emission_means_, self.emission_vars_)
            log_post = np.log(np.clip(self.startprob_, 1e-12, 1.0))[None, :] + log_em
            belief_today = np.exp(log_post - _logsumexp_np(log_post, axis=1, keepdims=True))

        if y_today is not None:
            oh_today = np.eye(self.n_states)[y_today]
            state_dist_t = 0.75 * oh_today + 0.25 * belief_today
        else:
            state_dist_t = belief_today

        # Paso predictivo de Markov: P_markov(S_{t+1}=k) = \sum_i P(S_t=i) A_{ik}
        markov_prior = np.clip(state_dist_t @ self.transmat_, 1e-12, 1.0)
        log_markov = self.markov_weight * np.log(markov_prior)

        # Verosimilitud generativa condicional meteorológica p(x_t | S_{t+1}=k)
        log_gen_lik = self.meteo_likelihood_temp * self._log_diag_gaussian(
            Xz, self.next_cond_means_, self.next_cond_vars_
        )
        log_gen_post = log_markov + log_gen_lik
        log_gen_post -= _logsumexp_np(log_gen_post, axis=1, keepdims=True)
        p_gen = np.exp(log_gen_post)

        # Componente log-lineal acoplado
        Phi = np.hstack([Xz, state_dist_t])
        logits_disc = Phi @ self.W_disc_ + self.b_disc_[None, :]
        p_disc = np.exp(logits_disc - _logsumexp_np(logits_disc, axis=1, keepdims=True))

        # Fusión Bayesiana en probabilidades calibradas
        p_final = (1.0 - self.disc_blend_) * p_gen + self.disc_blend_ * p_disc
        p_final /= np.sum(p_final, axis=1, keepdims=True)
        return p_final

    def predict_next_day(
        self,
        X: np.ndarray,
        y_today: Optional[np.ndarray] = None,
        station_groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.argmax(self.predict_next_day_proba(X, y_today=y_today, station_groups=station_groups), axis=1)

    def to_dict(self) -> Dict:
        return {
            "n_states": self.n_states,
            "trans_alpha": self.trans_alpha,
            "var_smoothing": self.var_smoothing,
            "markov_weight": self.markov_weight,
            "meteo_likelihood_temp": self.meteo_likelihood_temp,
            "disc_blend": self.disc_blend_,
            "startprob": self.startprob_.tolist(),
            "transmat": self.transmat_.tolist(),
            "emission_means": self.emission_means_.tolist(),
            "emission_vars": self.emission_vars_.tolist(),
            "next_cond_means": self.next_cond_means_.tolist(),
            "next_cond_vars": self.next_cond_vars_.tolist(),
            "feat_mean": self.feat_mean_.tolist(),
            "feat_std": self.feat_std_.tolist(),
            "W_disc": self.W_disc_.tolist(),
            "b_disc": self.b_disc_.tolist(),
            "feature_columns": FEATURE_COLUMNS,
            "state_labels": STATE_LABELS,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GaussianHMMScratch":
        obj = cls(
            n_states=data["n_states"],
            trans_alpha=data["trans_alpha"],
            var_smoothing=data["var_smoothing"],
            markov_weight=data["markov_weight"],
            meteo_likelihood_temp=data["meteo_likelihood_temp"],
        )
        obj.disc_blend_ = data["disc_blend"]
        obj.startprob_ = np.array(data["startprob"], dtype=float)
        obj.transmat_ = np.array(data["transmat"], dtype=float)
        obj.emission_means_ = np.array(data["emission_means"], dtype=float)
        obj.emission_vars_ = np.array(data["emission_vars"], dtype=float)
        obj.next_cond_means_ = np.array(data["next_cond_means"], dtype=float)
        obj.next_cond_vars_ = np.array(data["next_cond_vars"], dtype=float)
        obj.feat_mean_ = np.array(data["feat_mean"], dtype=float)
        obj.feat_std_ = np.array(data["feat_std"], dtype=float)
        obj.W_disc_ = np.array(data["W_disc"], dtype=float)
        obj.b_disc_ = np.array(data["b_disc"], dtype=float)
        return obj


def train_and_evaluate_classical_models(
    df: Optional[pd.DataFrame] = None,
    split_date: str = "2025-01-01",
    save_artifacts: bool = True,
) -> Dict:
    """
    Ejecuta el protocolo experimental completo de la Parte 1 con partición temporal estricta:
    - Entrenamiento: Periodo histórico anterior a `split_date` (Año 2024 completo).
    - Prueba (Hold-out futuro): Periodo igual o posterior a `split_date` (Año 2025 completo).
    Compara ambos modelos propios contra las 2 líneas base y persiste parámetros JSON.
    """
    if df is None:
        df = load_air_quality_data()

    work, X, y_today, y_tomorrow, station_base = prepare_features_and_targets(df, split_date=split_date)
    train_mask = (work["fecha"] < split_date).to_numpy()
    test_mask = ~train_mask

    X_train, y_today_tr, y_tom_tr = X[train_mask], y_today[train_mask], y_tomorrow[train_mask]
    X_test, y_today_te, y_tom_te = X[test_mask], y_today[test_mask], y_tomorrow[test_mask]
    st_train = work.loc[train_mask, "estacion"].to_numpy()
    st_test = work.loc[test_mask, "estacion"].to_numpy()

    # 1. Baseline Mayoritario
    maj = MajorityClassBaseline(n_classes=4).fit(y_tom_tr)
    pred_maj = maj.predict(len(y_tom_te))
    prob_maj = maj.predict_proba(len(y_tom_te))
    metrics_maj = compute_classification_metrics_np(y_tom_te, pred_maj, prob_maj)

    # 2. Baseline Persistencia Temporal (y_{t+1} = y_t)
    pers = PersistenceBaseline(n_classes=4)
    pred_pers = pers.predict(y_today_te)
    prob_pers = pers.predict_proba(y_today_te)
    metrics_pers = compute_classification_metrics_np(y_tom_te, pred_pers, prob_pers)

    # 3. Modelo Propio 1: Gaussian Naive Bayes From Scratch
    gnb = GaussianNaiveBayesScratch(n_classes=4).fit(X_train, y_tom_tr)
    prob_gnb = gnb.predict_proba(X_test)
    pred_gnb = np.argmax(prob_gnb, axis=1)
    metrics_gnb = compute_classification_metrics_np(y_tom_te, pred_gnb, prob_gnb)

    # 4. Modelo Propio 2: Gaussian HMM + Dynamic Bayesian Predictor From Scratch
    hmm = GaussianHMMScratch(n_states=4).fit(X_train, y_today_tr, y_tom_tr, station_groups=st_train)
    prob_hmm = hmm.predict_next_day_proba(X_test, y_today=y_today_te, station_groups=st_test)
    pred_hmm = np.argmax(prob_hmm, axis=1)
    metrics_hmm = compute_classification_metrics_np(y_tom_te, pred_hmm, prob_hmm)

    # Estadísticas de referencia por estación para inferencia puntual desde herramientas del agente
    station_stats = {}
    for st_name in STATIONS_METADATA:
        sub_st = df[(df["estacion"] == st_name) & (df["fecha"] < split_date)]
        if sub_st.empty:
            sub_st = df[df["estacion"] == st_name]
        station_stats[st_name] = {
            "temperatura_mediana": float(sub_st["temperatura"].median()),
            "humedad_mediana": float(sub_st["humedad_relativa"].median()),
            "viento_mediana": float(sub_st["velocidad_viento"].median()),
            "pm25_mediana": float(sub_st["pm25"].median()),
            "pm25_base_train": float(station_base.get(st_name, sub_st["pm25"].mean())),
        }

    summary = {
        "split_date": split_date,
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
        "models": {
            "Baseline Frecuencia Mayoritaria": metrics_maj,
            "Baseline Persistencia (y_t = y_{t-1})": metrics_pers,
            "Gaussian Naive Bayes (Propio - NumPy)": metrics_gnb,
            "Gaussian HMM + Filtro Bayesiano (Propio - NumPy)": metrics_hmm,
        },
        "hmm_params": hmm.to_dict(),
        "gnb_params": {
            "priors": gnb.priors_.tolist(),
            "means": gnb.means_.tolist(),
            "vars": gnb.vars_.tolist(),
            "feat_mean": gnb.feat_mean_.tolist(),
            "feat_std": gnb.feat_std_.tolist(),
            "likelihood_temp": gnb.likelihood_temp,
            "prior_weight": gnb.prior_weight,
        },
        "station_stats": station_stats,
    }

    if save_artifacts:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_ARTIFACTS_JSON.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return summary


_CACHED_ARTIFACTS: Optional[Dict] = None


def load_or_train_classical_model() -> Tuple[GaussianHMMScratch, Dict]:
    """Carga el modelo clásico entrenado desde disco (o lo entrena automáticamente si no existe)."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is None:
        if not MODEL_ARTIFACTS_JSON.exists():
            _CACHED_ARTIFACTS = train_and_evaluate_classical_models(save_artifacts=True)
        else:
            _CACHED_ARTIFACTS = json.loads(MODEL_ARTIFACTS_JSON.read_text(encoding="utf-8"))
    hmm = GaussianHMMScratch.from_dict(_CACHED_ARTIFACTS["hmm_params"])
    return hmm, _CACHED_ARTIFACTS


def predict_station_air_quality(
    estacion: str,
    viento: float,
    humedad: float,
    pm25_previo: float,
    temperatura: Optional[float] = None,
    precipitacion: float = 0.0,
    es_periodo_poeca: int = 1,
) -> Dict:
    """
    Interfaz limpia para invocar el modelo clásico de la Parte 1 desde las herramientas del agente
    y desde el simulador del Dashboard Streamlit.
    """
    canonical_station = resolve_station_name(estacion)
    if viento < 0 or viento > 25:
        raise ValueError(f"Velocidad del viento fuera de rango físico válido (0 a 25 m/s): {viento}")
    if humedad < 5 or humedad > 100:
        raise ValueError(f"Humedad relativa fuera de rango válido (5% a 100%): {humedad}")
    if pm25_previo < 0 or pm25_previo > 500:
        raise ValueError(f"PM2.5 previo fuera de rango válido (0 a 500 ug/m3): {pm25_previo}")

    hmm, artifacts = load_or_train_classical_model()
    st_defaults = artifacts["station_stats"].get(canonical_station, {})
    temp_val = float(temperatura) if temperatura is not None else st_defaults.get("temperatura_mediana", 22.5)

    # Estimación física suave del estado actual a partir del PM2.5 previo y anomalías de ventilación
    wind_anom = (st_defaults.get("viento_mediana", 1.2) - float(viento)) * 3.2
    hum_anom = (float(humedad) - st_defaults.get("humedad_mediana", 62.0)) * 0.12
    rain_scav = -min(float(precipitacion), 25.0) * 0.28
    pm25_est_hoy = float(np.clip(pm25_previo + wind_anom + hum_anom + rain_scav, 2.0, 250.0))

    # Categoría base actual según umbrales Resolución 2254
    if pm25_previo <= 12.0:
        cur_state = 0
    elif pm25_previo <= 37.0:
        cur_state = 1
    elif pm25_previo <= 55.4:
        cur_state = 2
    else:
        cur_state = 3

    base_est = float(st_defaults.get("pm25_base_train", st_defaults.get("pm25_mediana", 18.0)))
    x_vec = np.array(
        [[
            pm25_est_hoy,
            float(pm25_previo),
            0.5 * (pm25_est_hoy + float(pm25_previo)),
            pm25_est_hoy - float(pm25_previo),
            base_est,
            pm25_est_hoy - base_est,
            float(viento),
            float(humedad),
            temp_val,
            float(precipitacion),
            float(es_periodo_poeca),
        ]],
        dtype=float,
    )

    probs = hmm.predict_next_day_proba(x_vec, y_today=np.array([cur_state]))[0]
    pred_idx = int(np.argmax(probs))
    pred_cat = STATE_LABELS[pred_idx]
    pred_color = CATEGORY_TO_COLOR[pred_cat]

    # Nivel operativo POECA asociado
    poeca_map = {
        "Verde": "Nivel Normal (Vigilancia Continua)",
        "Amarillo": "Nivel Normal (Seguimiento Preventivo)",
        "Naranja": "Estado de Prevención POECA (Nivel I - Grupos Sensibles)",
        "Rojo": "Estado de Alerta POECA (Nivel II - Restricción Vehicular e Industrial)",
    }

    return {
        "estacion": canonical_station,
        "modelo": "Gaussian HMM + Filtro Bayesiano Dinámico (Implementación Propia en NumPy)",
        "categoria_predicha": pred_cat,
        "color_ica_predicho": pred_color,
        "nivel_poeca_asociado": poeca_map.get(pred_color, "Vigilancia"),
        "confianza": round(float(probs[pred_idx]), 4),
        "pm25_estimado_ugm3": round(pm25_est_hoy, 1),
        "probabilidades_por_estado": {
            f"{STATE_LABELS[i]} ({CATEGORY_TO_COLOR[STATE_LABELS[i]]})": round(float(probs[i]), 4)
            for i in range(len(STATE_LABELS))
        },
        "parametros_entrada": {
            "viento_ms": round(float(viento), 2),
            "humedad_pct": round(float(humedad), 2),
            "pm25_previo_ugm3": round(float(pm25_previo), 2),
            "temperatura_c": round(temp_val, 2),
            "precipitacion_mm": round(float(precipitacion), 2),
        },
        "fuente_modelo": "Modelo Probabilístico Parte 1 entrenado con histórico SISAIRE-AMVA (2024)",
    }


if __name__ == "__main__":
    res = train_and_evaluate_classical_models(save_artifacts=True)
    for name, m in res["models"].items():
        print(f"{name:48s} | Acc={m['accuracy']:.4f} | Macro-F1={m['macro_f1']:.4f} | Brier={m['brier_score']}")
