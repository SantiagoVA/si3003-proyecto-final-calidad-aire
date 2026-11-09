"""
Definición de Herramientas Operativas (Tool Calling) para el Agente Ambiental del Valle de Aburrá.

Implementa las 3 herramientas especificadas en `PLAN_DE_TRABAJO.md`:
1. `get_air_quality_data(estacion, fecha=None)`: Consulta mediciones reales históricas o recientes
   de $PM_{2.5}$, ICA y meteorología de las estaciones SIATA/AMVA (SISAIRE).
2. `predict_air_quality(estacion, viento, humedad, pm25_previo, temperatura=None, precipitacion=0.0)`:
   Invoca el modelo probabilístico clásico propio de la Parte 1 (`GaussianHMMScratch` en NumPy)
   y retorna categoría ICA predicha, nivel POECA y distribución de probabilidad posterior.
3. `search_poeca_protocols(consulta, top_k=3)`: Recupera fragmentos normativos y de salud pública
   desde el índice vectorial FAISS con citas bibliográficas verificables.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.classical_model import predict_station_air_quality
from src.data_loader import (
    STATIONS_METADATA,
    load_air_quality_data,
    resolve_station_name,
)
from src.rag_engine import get_rag_engine


def get_air_quality_data(estacion: str, fecha: Optional[str] = None) -> Dict[str, Any]:
    """
    Consulta mediciones reales de PM2.5, ICA y variables meteorológicas de una estación
    del Valle de Aburrá (o de 'todas' para resumen regional) en una fecha específica
    o en la fecha más reciente disponible.
    """
    df = load_air_quality_data()
    min_date, max_date = str(df["fecha"].min()), str(df["fecha"].max())

    target_date = max_date
    if fecha is not None and str(fecha).strip().lower() not in ("", "none", "null", "hoy", "actual", "reciente"):
        clean_date = str(fecha).strip()[:10]
        try:
            pd.to_datetime(clean_date, format="%Y-%m-%d", errors="raise")
        except Exception as exc:
            raise ValueError(
                f"Formato de fecha inválido '{fecha}'. Use formato ISO 'YYYY-MM-DD' "
                f"dentro del rango disponible ({min_date} a {max_date})."
            ) from exc
        if clean_date < min_date or clean_date > max_date:
            raise ValueError(
                f"La fecha '{clean_date}' está fuera del rango histórico consolidado "
                f"({min_date} a {max_date})."
            )
        target_date = clean_date

    # Permitir consulta panorámica de todas las estaciones del Valle de Aburrá
    if estacion and str(estacion).strip().lower() in ("todas", "valle de aburra", "valle de aburrá", "general", "red"):
        sub_all = df[df["fecha"] == target_date].sort_values("pm25", ascending=False)
        resumen = []
        for _, row in sub_all.iterrows():
            resumen.append(
                {
                    "estacion": row["estacion"],
                    "municipio": row["municipio"],
                    "pm25_ugm3": float(row["pm25"]),
                    "ica": float(row["ica"]),
                    "categoria_ica": row["categoria_ica"],
                    "color_ica": row["color_ica"],
                    "viento_ms": float(row["velocidad_viento"]),
                    "humedad_pct": float(row["humedad_relativa"]),
                    "temperatura_c": float(row["temperatura"]),
                }
            )
        return {
            "modo_consulta": "Resumen Regional Valle de Aburrá (5 estaciones)",
            "fecha": target_date,
            "estaciones": resumen,
            "estacion_mas_critica": resumen[0]["estacion"] if resumen else None,
            "pm25_promedio_valle": round(float(sub_all["pm25"].mean()), 2),
            "fuente": (
                f"[Fuente: Datos Abiertos IDEAM-SISAIRE / Red SIATA-AMVA (Recurso g4t8-zkc3), "
                f"Resumen Regional Valle de Aburrá, Fecha {target_date}]"
            ),
        }

    canonical = resolve_station_name(estacion)
    sub = df[(df["estacion"] == canonical) & (df["fecha"] == target_date)]
    if sub.empty:
        raise ValueError(f"No se encontraron datos para la estación '{canonical}' en la fecha '{target_date}'.")

    row = sub.iloc[0]
    hist_7d = df[(df["estacion"] == canonical) & (df["fecha"] <= target_date)].tail(7)

    return {
        "estacion": canonical,
        "estacion_oficial": row["estacion_oficial"],
        "municipio": row["municipio"],
        "tipo_estacion": row["tipo_estacion"],
        "fecha": target_date,
        "pm25_24h_ugm3": float(row["pm25"]),
        "pm25_max_horario_ugm3": float(row["pm25_max_horario"]),
        "pm25_previo_ugm3": float(row["pm25_previo"]),
        "media_movil_7d_ugm3": round(float(hist_7d["pm25"].mean()), 2),
        "ica": float(row["ica"]),
        "categoria_ica": str(row["categoria_ica"]),
        "color_ica": str(row["color_ica"]),
        "temperatura_c": float(row["temperatura"]),
        "humedad_relativa_pct": float(row["humedad_relativa"]),
        "velocidad_viento_ms": float(row["velocidad_viento"]),
        "precipitacion_mm": float(row["precipitacion"]),
        "periodo_estacional": str(row["periodo_poeca"]),
        "supera_norma_diaria_37ugm3": bool(float(row["pm25"]) > 37.0),
        "fuente": (
            f"[Fuente: Datos Abiertos IDEAM-SISAIRE / Red SIATA-AMVA (Recurso g4t8-zkc3), "
            f"Estación {row['estacion_oficial']}, Fecha {target_date}]"
        ),
    }


def predict_air_quality(
    estacion: str,
    viento: float,
    humedad: float,
    pm25_previo: float,
    temperatura: Optional[float] = None,
    precipitacion: float = 0.0,
) -> Dict[str, Any]:
    """
    Ejecuta el modelo probabilístico clásico implementado desde cero en la Parte 1
    (`GaussianHMMScratch` + Filtro Bayesiano Dinámico en NumPy) para pronosticar
    el estado de calidad del aire del día siguiente bajo condiciones meteorológicas dadas.
    """
    try:
        viento_f = float(viento)
        humedad_f = float(humedad)
        pm25_f = float(pm25_previo)
        temp_f = float(temperatura) if temperatura is not None else None
        rain_f = float(precipitacion) if precipitacion is not None else 0.0
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Los argumentos `viento`, `humedad` y `pm25_previo` deben ser valores numéricos reales."
        ) from exc

    pred = predict_station_air_quality(
        estacion=estacion,
        viento=viento_f,
        humedad=humedad_f,
        pm25_previo=pm25_f,
        temperatura=temp_f,
        precipitacion=rain_f,
    )
    pred["fuente"] = (
        f"[Fuente: Modelo Probabilístico Clásico Propio — Gaussian HMM + Filtro Bayesiano (Parte 1, NumPy), "
        f"entrenado con histórico SISAIRE-AMVA para Estación {pred['estacion']}]"
    )
    return pred


def search_poeca_protocols(consulta: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Consulta el índice vectorial semántico FAISS sobre el Protocolo POECA (AMVA),
    la Resolución 2254 de 2017 (MinAmbiente) y las Guías de Salud Pública del SIATA.
    """
    rag = get_rag_engine()
    k = max(1, min(int(top_k), 5))
    results = rag.search(query=consulta, top_k=k)
    return {
        "consulta": consulta,
        "fragmentos_recuperados": len(results),
        "resultados": results,
        "citas_obligatorias": [r["cita"] for r in results],
    }


TOOLS_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_air_quality_data": get_air_quality_data,
    "predict_air_quality": predict_air_quality,
    "search_poeca_protocols": search_poeca_protocols,
}

TOOLS_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_air_quality_data",
            "description": (
                "Consulta mediciones reales históricas o actuales de PM2.5 (ug/m3), Índice ICA, "
                "temperatura, humedad relativa, viento y precipitación para una estación del Valle "
                "de Aburrá ('Centro', 'El Poblado', 'Aranjuez', 'Itagüí', 'Bello' o 'todas')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {
                        "type": "string",
                        "description": (
                            "Nombre de la estación ('Centro', 'El Poblado', 'Aranjuez', 'Itagüí', 'Bello' "
                            "o 'todas' para resumen general del Valle de Aburrá)."
                        ),
                    },
                    "fecha": {
                        "type": "string",
                        "description": (
                            "Fecha opcional en formato 'YYYY-MM-DD' (rango 2024-01-01 a 2025-12-31). "
                            "Omitir o pasar null para consultar la medición más reciente."
                        ),
                    },
                },
                "required": ["estacion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_air_quality",
            "description": (
                "Ejecuta el modelo probabilístico clásico propio de la Parte 1 (Gaussian HMM + Filtro Bayesiano "
                "implementado en NumPy) para predecir la categoría ICA del día siguiente y sus probabilidades."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {
                        "type": "string",
                        "description": "Estación objetivo ('Centro', 'El Poblado', 'Aranjuez', 'Itagüí', 'Bello').",
                    },
                    "viento": {
                        "type": "number",
                        "description": "Velocidad del viento proyectada o actual en m/s (ej. 0.6 a 3.5).",
                    },
                    "humedad": {
                        "type": "number",
                        "description": "Humedad relativa en porcentaje % (ej. 40 a 90).",
                    },
                    "pm25_previo": {
                        "type": "number",
                        "description": "Concentración previa/actual de PM2.5 en ug/m3 (ej. 10 a 75).",
                    },
                    "temperatura": {
                        "type": "number",
                        "description": "Temperatura opcional en °C (ej. 22.0).",
                    },
                    "precipitacion": {
                        "type": "number",
                        "description": "Precipitación diaria opcional en mm (por defecto 0.0).",
                    },
                },
                "required": ["estacion", "viento", "humedad", "pm25_previo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_poeca_protocols",
            "description": (
                "Busca en la base vectorial RAG normativa del Protocolo POECA (AMVA), Resolución 2254 de 2017 "
                "(MinAmbiente), guías de salud/deporte según colores del ICA y dinámica meteorológica del SIATA."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Pregunta o términos clave normativos, de salud o meteorológicos a consultar.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Cantidad de fragmentos normativos a recuperar (por defecto 3).",
                    },
                },
                "required": ["consulta"],
            },
        },
    },
]

