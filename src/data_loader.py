"""
Módulo de Adquisición, Saneamiento y Carga de Datos Reales (SIATA / AMVA - SISAIRE IDEAM).

Fuente Oficial Verificada:
- Portal de Datos Abiertos de Colombia (datos.gov.co):
  1. Recurso `g4t8-zkc3`: "Calidad del Aire en Colombia" (IDEAM - Subsistema SISAIRE),
     registros horarios reportados por la autoridad ambiental AMVA (Red SIATA) 2024-2025.
  2. Recurso `kekd-7v7h`: "Calidad del Aire en Colombia (Promedio Anual)" (IDEAM - SISAIRE).

Implementa Plan B por diseño:
- Si el CSV procesado local (`data/processed/air_quality_aburra.csv`) existe, lo carga de
  forma instantánea garantizando 100% de reproducibilidad sin depender de disponibilidad de red.
- Incluye el pipeline completo `download_and_build_dataset()` para auditar y reconstruir los
  datos desde la API Socrata oficial de `datos.gov.co`.
"""

from __future__ import annotations

import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CSV = PROCESSED_DIR / "air_quality_aburra.csv"
HOURLY_EPISODE_CSV = RAW_DIR / "sisaire_amva_hourly_episodio_marzo2024.csv"
RAW_DAILY_CSV = RAW_DIR / "sisaire_amva_daily_raw_2024_2025.csv"
ANNUAL_REF_CSV = RAW_DIR / "sisaire_amva_anual_kekd_7v7h.csv"
MANIFEST_JSON = RAW_DIR / "manifest_descarga_sisaire.json"

SOCRATA_HOURLY_URL = "https://www.datos.gov.co/resource/g4t8-zkc3.json"
SOCRATA_ANNUAL_URL = "https://www.datos.gov.co/resource/kekd-7v7h.json"

# Estaciones representativas del Valle de Aburrá (AMVA / SIATA en SISAIRE)
STATIONS_METADATA: Dict[str, Dict] = {
    "Centro": {
        "nombre_corto": "Centro",
        "nombre_oficial": "Medellín Centro (Tráfico Centro - Museo de Antioquia)",
        "municipio": "Medellín",
        "zona": "Centro - La Candelaria (ZUAP)",
        "tipo": "Tráfico Urbano",
        "latitud": 6.252709,
        "longitud": -75.569656,
        "pm25_ids": ["29509"],  # TRÁFICO CENTRO
        "meteo_ids": ["31994", "31852", "31854"],  # EPM, Miraflores, Caribe
        "aliases": [
            "centro", "medellin centro", "trafico centro", "museo de antioquia",
            "la candelaria", "zuap", "med-cen", "estacion centro"
        ],
    },
    "El Poblado": {
        "nombre_corto": "El Poblado",
        "nombre_oficial": "El Poblado (I.E. José Félix de Restrepo - Suroriente)",
        "municipio": "Medellín",
        "zona": "Suroriente - Comuna 14",
        "tipo": "Representatividad Poblacional (Residencial)",
        "latitud": 6.199869,
        "longitud": -75.560835,
        "pm25_ids": ["31373", "30732", "32034"],  # I.E. José Félix, Tanques La Ye, U. CES
        "meteo_ids": ["30732", "32034", "31852"],  # Tanques La Ye, U. CES, Miraflores
        "aliases": [
            "el poblado", "poblado", "jose felix", "universidad ces", "la ye",
            "medellin sur", "estacion poblado", "estacion el poblado"
        ],
    },
    "Aranjuez": {
        "nombre_corto": "Aranjuez",
        "nombre_oficial": "Aranjuez (I.E. Ciro Mendía - Nororiente Medellín)",
        "municipio": "Medellín",
        "zona": "Nororiente - Comuna 4",
        "tipo": "Representatividad Poblacional (Urbana)",
        "latitud": 6.290140,
        "longitud": -75.555779,
        "pm25_ids": ["31364", "31946"],  # I.E. Ciro Mendía, Biblioteca La Quintana
        "meteo_ids": ["31946", "31854"],  # La Quintana, Fiscalía Caribe
        "aliases": [
            "aranjuez", "ciro mendia", "medellin norte", "nororiente",
            "estacion aranjuez", "comuna 4"
        ],
    },
    "Itagüí": {
        "nombre_corto": "Itagüí",
        "nombre_oficial": "Itagüí (Casa de Justicia - Corredor Sur Industrial)",
        "municipio": "Itagüí",
        "zona": "Sur del Valle de Aburrá",
        "tipo": "Representatividad Poblacional / Industrial-Urbana",
        "latitud": 6.185667,
        "longitud": -75.597206,
        "pm25_ids": ["29493", "29511"],  # Casa Justicia, Concejo Municipal Itagüí
        "meteo_ids": ["29493", "29511", "31628"],  # Casa Justicia (co-localizada), Concejo, Sabaneta
        "aliases": [
            "itagui", "itagüí", "casa justicia", "casa de justicia",
            "sur industrial", "concejo itagui", "estacion itagui"
        ],
    },
    "Bello": {
        "nombre_corto": "Bello",
        "nombre_oficial": "Bello (Liceo Fernando Vélez - Corredor Norte)",
        "municipio": "Bello",
        "zona": "Norte del Valle de Aburrá",
        "tipo": "Representatividad Poblacional (Urbana Norte)",
        "latitud": 6.337377,
        "longitud": -75.567748,
        "pm25_ids": ["31362", "31360"],  # Liceo Fernando Vélez, Copacabana
        "meteo_ids": ["29502", "31360"],  # U. San Buenaventura Bello, Copacabana
        "aliases": [
            "bello", "fernando velez", "liceo fernando velez",
            "corredor norte", "estacion bello"
        ],
    },
}

# Tabla oficial de puntos de corte ICA para PM2.5 (24h) - Resolución 2254 de 2017 (MinAmbiente)
ICA_BREAKPOINTS_PM25: List[Tuple[float, float, int, int, str, str, str]] = [
    (0.0, 12.0, 0, 50, "Buena", "Verde", "#2ecc71"),
    (12.1, 37.0, 51, 100, "Aceptable", "Amarillo", "#f1c40f"),
    (37.1, 55.4, 101, 150, "Dañina a Grupos Sensibles", "Naranja", "#e67e22"),
    (55.5, 150.4, 151, 200, "Dañina a la Salud", "Rojo", "#e74c3c"),
    (150.5, 250.4, 201, 300, "Muy Dañina a la Salud", "Púrpura", "#8e44ad"),
    (250.5, 500.4, 301, 500, "Peligrosa", "Marrón", "#6e2c00"),
]

CATEGORY_ORDER = [
    "Buena",
    "Aceptable",
    "Dañina a Grupos Sensibles",
    "Dañina a la Salud",
]

COLOR_MAP_HEX = {
    "Verde": "#2ecc71",
    "Amarillo": "#f1c40f",
    "Naranja": "#e67e22",
    "Rojo": "#e74c3c",
    "Púrpura": "#8e44ad",
    "Marrón": "#6e2c00",
}

CATEGORY_TO_COLOR = {
    "Buena": "Verde",
    "Aceptable": "Amarillo",
    "Dañina a Grupos Sensibles": "Naranja",
    "Dañina a la Salud": "Rojo",
    "Muy Dañina a la Salud": "Púrpura",
    "Peligrosa": "Marrón",
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower().strip())
        if unicodedata.category(c) != "Mn"
    )


def resolve_station_name(query_station: str) -> str:
    """
    Normaliza cualquier variante de nombre de estación escrita por el usuario o el LLM.
    Lanza ValueError descriptivo si la estación no pertenece a la red configurada.
    """
    if not query_station or not isinstance(query_station, str):
        raise ValueError(
            f"Nombre de estación vacío o inválido. Estaciones válidas: {list(STATIONS_METADATA.keys())}"
        )
    norm = _strip_accents(query_station)
    for canonical, meta in STATIONS_METADATA.items():
        if norm == _strip_accents(canonical):
            return canonical
        for alias in meta["aliases"]:
            if norm == _strip_accents(alias) or _strip_accents(alias) in norm:
                return canonical
    raise ValueError(
        f"Estación '{query_station}' no reconocida en el Valle de Aburrá. "
        f"Estaciones disponibles: {', '.join(STATIONS_METADATA.keys())}."
    )


def calculate_ica_pm25(pm25_24h: float) -> Tuple[float, str, str, str]:
    """
    Calcula el Índice de Calidad del Aire (ICA) oficial según el Artículo 19 y Tabla 6
    de la Resolución 2254 de 2017 (Ministerio de Ambiente y Desarrollo Sostenible):
        I_p = ((I_Hi - I_Lo) / (BP_Hi - BP_Lo)) * (C_p - BP_Lo) + I_Lo
    """
    if pd.isna(pm25_24h):
        return np.nan, "Sin Dato", "Gris", "#95a5a6"
    cp = round(max(0.0, float(pm25_24h)), 1)
    for bp_lo, bp_hi, i_lo, i_hi, categoria, color, hex_code in ICA_BREAKPOINTS_PM25:
        if bp_lo <= cp <= bp_hi:
            ica = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + i_lo
            return round(ica, 1), categoria, color, hex_code
    # Si supera el máximo de la tabla (500.4 ug/m3)
    return 500.0, "Peligrosa", "Marrón", "#6e2c00"


def _fetch_month_socrata(
    year: int, month: int, all_station_ids: List[str]
) -> Tuple[str, List[dict]]:
    """Descarga un mes completo de registros horarios de las estaciones objetivo desde datos.gov.co."""
    start_dt = f"{year:04d}-{month:02d}-01 00:00:00"
    if month == 12:
        end_dt = f"{year + 1:04d}-01-01 00:00:00"
    else:
        end_dt = f"{year:04d}-{month + 1:02d}-01 00:00:00"

    ids_sql = "(" + ",".join(f"'{sid}'" for sid in sorted(set(all_station_ids))) + ")"
    vars_sql = "('PM2.5','TAire2','HAire2','VViento','PLiquida')"

    rows: List[dict] = []
    offset = 0
    page_size = 50000
    while True:
        params = {
            "$select": (
                "estacion_id,nombre_est,municipio,msfl_code,"
                "med_concentracion_estandar,med_fecha_inicio,latitud,longitud"
            ),
            "$where": (
                f"nombre_fgda='AMVA' AND estacion_id in {ids_sql} "
                f"AND msfl_code in {vars_sql} "
                f"AND med_fecha_inicio >= '{start_dt}' AND med_fecha_inicio < '{end_dt}'"
            ),
            "$order": "med_fecha_inicio ASC",
            "$limit": page_size,
            "$offset": offset,
        }
        resp = requests.get(SOCRATA_HOURLY_URL, params=params, timeout=40)
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    return f"{year:04d}-{month:02d}", rows


def download_and_build_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Descarga los datos horarios oficiales 2024-2025 desde `datos.gov.co` (SISAIRE-AMVA),
    ejecuta validaciones de rango físico, consolida promedios diarios de 24h conforme a la
    Resolución 2254 de 2017 y construye las variables meteorológicas y rezagos para la Parte 1.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if PROCESSED_CSV.exists() and not force_download:
        return load_air_quality_data()

    all_ids: List[str] = []
    for meta in STATIONS_METADATA.values():
        all_ids.extend(meta["pm25_ids"])
        all_ids.extend(meta["meteo_ids"])
    all_ids = sorted(set(all_ids))

    # 1. Descargar referencia anual oficial de kekd-7v7h para trazabilidad
    try:
        r_ann = requests.get(
            SOCRATA_ANNUAL_URL,
            params={
                "$where": "autoridad_ambiental='AMVA' AND variable='PM2.5' AND a_o >= '2021'",
                "$limit": 500,
            },
            timeout=25,
        )
        if r_ann.status_code == 200 and r_ann.json():
            pd.DataFrame(r_ann.json()).to_csv(ANNUAL_REF_CSV, index=False)
    except Exception as exc:
        print(f"[Aviso] No se pudo guardar referencia anual opcional: {exc}")

    # 2. Descarga paralela de los 24 meses (2024-01 a 2025-12) desde g4t8-zkc3
    months = [(yr, mo) for yr in (2024, 2025) for mo in range(1, 13)]
    monthly_counts: Dict[str, int] = {}
    all_raw_rows: List[dict] = []

    print(f"Descargando registros horarios reales SISAIRE-AMVA (2024-2025) para {len(all_ids)} IDs...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_month_socrata, yr, mo, all_ids): (yr, mo)
            for yr, mo in months
        }
        for fut in as_completed(futures):
            ym, batch = fut.result()
            monthly_counts[ym] = len(batch)
            all_raw_rows.extend(batch)
            print(f"  -> Mes {ym}: {len(batch):,} registros horarios descargados")

    df_raw = pd.DataFrame(all_raw_rows)
    df_raw["valor"] = pd.to_numeric(df_raw["med_concentracion_estandar"], errors="coerce")
    df_raw["fecha_hora"] = pd.to_datetime(df_raw["med_fecha_inicio"], errors="coerce")
    df_raw = df_raw.dropna(subset=["fecha_hora", "valor"]).copy()
    df_raw["fecha"] = df_raw["fecha_hora"].dt.strftime("%Y-%m-%d")
    df_raw["hora"] = df_raw["fecha_hora"].dt.hour

    # Guardar ventana horaria cruda del episodio crítico de marzo 2024 para EDA intradiario
    df_mar2024 = df_raw[
        (df_raw["fecha"] >= "2024-02-25") & (df_raw["fecha"] <= "2024-04-05")
    ].sort_values(["fecha_hora", "estacion_id", "msfl_code"])
    df_mar2024.to_csv(HOURLY_EPISODE_CSV, index=False)

    # 3. Saneamiento físico de rangos horarios según protocolo IDEAM
    bounds = {
        "PM2.5": (0.5, 300.0),
        "TAire2": (8.0, 38.0),
        "HAire2": (15.0, 100.0),
        "VViento": (0.0, 18.0),
        "PLiquida": (0.0, 120.0),
    }
    valid_mask = pd.Series(False, index=df_raw.index)
    for code, (lo, hi) in bounds.items():
        valid_mask |= (df_raw["msfl_code"] == code) & df_raw["valor"].between(lo, hi)
    df_clean_h = df_raw[valid_mask].copy()

    # Agregación diaria por estacion_id y variable
    daily_by_id = (
        df_clean_h.groupby(["fecha", "estacion_id", "nombre_est", "msfl_code"], as_index=False)
        .agg(
            promedio=("valor", "mean"),
            maximo=("valor", "max"),
            minimo=("valor", "min"),
            suma=("valor", "sum"),
            n_horas=("valor", "count"),
        )
    )
    daily_by_id.to_csv(RAW_DAILY_CSV, index=False)

    # Mapa rápido (fecha, estacion_id, msfl_code) -> métricas
    piv_mean = daily_by_id.pivot_table(
        index="fecha", columns=["estacion_id", "msfl_code"], values="promedio"
    )
    piv_max = daily_by_id.pivot_table(
        index="fecha", columns=["estacion_id", "msfl_code"], values="maximo"
    )
    piv_min = daily_by_id.pivot_table(
        index="fecha", columns=["estacion_id", "msfl_code"], values="minimo"
    )
    piv_sum = daily_by_id.pivot_table(
        index="fecha", columns=["estacion_id", "msfl_code"], values="suma"
    )
    piv_cnt = daily_by_id.pivot_table(
        index="fecha", columns=["estacion_id", "msfl_code"], values="n_horas"
    )

    full_dates = pd.date_range("2024-01-01", "2025-12-31", freq="D").strftime("%Y-%m-%d")
    piv_mean = piv_mean.reindex(full_dates)
    piv_max = piv_max.reindex(full_dates)
    piv_min = piv_min.reindex(full_dates)
    piv_sum = piv_sum.reindex(full_dates)
    piv_cnt = piv_cnt.reindex(full_dates)

    def _first_available_series(piv: pd.DataFrame, ids: List[str], var: str) -> pd.Series:
        out = pd.Series(np.nan, index=full_dates, dtype=float)
        for sid in ids:
            if (sid, var) in piv.columns:
                out = out.combine_first(piv[(sid, var)])
        return out

    # Promedio regional Valle de Aburrá por día (para imputar huecos puntuales de sensores)
    regional_vars = {}
    for var in ["PM2.5", "TAire2", "HAire2", "VViento", "PLiquida"]:
        matching_cols = [c for c in (piv_sum if var == "PLiquida" else piv_mean).columns if c[1] == var]
        if matching_cols:
            src_df = piv_sum[matching_cols] if var == "PLiquida" else piv_mean[matching_cols]
            regional_vars[var] = src_df.mean(axis=1)
        else:
            regional_vars[var] = pd.Series(0.0, index=full_dates)

    station_frames: List[pd.DataFrame] = []
    for st_key, meta in STATIONS_METADATA.items():
        pm_mean = _first_available_series(piv_mean, meta["pm25_ids"], "PM2.5")
        pm_max = _first_available_series(piv_max, meta["pm25_ids"], "PM2.5")
        pm_min = _first_available_series(piv_min, meta["pm25_ids"], "PM2.5")
        pm_hours = _first_available_series(piv_cnt, meta["pm25_ids"], "PM2.5")

        temp = _first_available_series(piv_mean, meta["meteo_ids"], "TAire2")
        hum = _first_available_series(piv_mean, meta["meteo_ids"], "HAire2")
        wind = _first_available_series(piv_mean, meta["meteo_ids"], "VViento")
        rain = _first_available_series(piv_sum, meta["meteo_ids"], "PLiquida")

        # Marcar si el dato de PM2.5 fue observado directamente antes de interpolación corta
        observado_directo = pm_mean.notna()

        # Imputación conservadora: interpolación temporal corta (máx 3 días) + ajuste regional
        ratio_est = (pm_mean / regional_vars["PM2.5"]).median()
        if pd.isna(ratio_est) or ratio_est <= 0:
            ratio_est = 1.0
        pm_filled = (
            pm_mean.interpolate(method="linear", limit=3)
            .combine_first(regional_vars["PM2.5"] * ratio_est)
            .bfill()
            .ffill()
        )
        pm_max_filled = pm_max.combine_first(pm_filled * 1.45)
        pm_min_filled = pm_min.combine_first(pm_filled * 0.55)
        temp_filled = temp.interpolate(limit=3).combine_first(regional_vars["TAire2"]).bfill().ffill()
        hum_filled = hum.interpolate(limit=3).combine_first(regional_vars["HAire2"]).bfill().ffill()
        wind_filled = wind.interpolate(limit=3).combine_first(regional_vars["VViento"]).bfill().ffill()
        rain_filled = rain.combine_first(regional_vars["PLiquida"]).fillna(0.0).clip(lower=0.0)

        st_df = pd.DataFrame(
            {
                "fecha": full_dates,
                "estacion": st_key,
                "estacion_oficial": meta["nombre_oficial"],
                "municipio": meta["municipio"],
                "zona": meta["zona"],
                "tipo_estacion": meta["tipo"],
                "latitud": meta["latitud"],
                "longitud": meta["longitud"],
                "pm25": pm_filled.round(2).values,
                "pm25_max_horario": pm_max_filled.round(2).values,
                "pm25_min_horario": pm_min_filled.round(2).values,
                "horas_validas": pm_hours.fillna(0).astype(int).values,
                "dato_observado": observado_directo.values,
                "temperatura": temp_filled.round(2).values,
                "humedad_relativa": hum_filled.round(2).values,
                "velocidad_viento": wind_filled.round(2).values,
                "precipitacion": rain_filled.round(2).values,
            }
        )

        # Cálculo del ICA y categoría normativa Resolución 2254 de 2017
        ica_tuples = [calculate_ica_pm25(v) for v in st_df["pm25"]]
        st_df["ica"] = [t[0] for t in ica_tuples]
        st_df["categoria_ica"] = [t[1] for t in ica_tuples]
        st_df["color_ica"] = [t[2] for t in ica_tuples]

        # Variables de rezago y medias móviles por estación (sin fuga de información futura)
        st_df["pm25_previo"] = st_df["pm25"].shift(1).bfill().round(2)
        st_df["pm25_ma3"] = st_df["pm25"].rolling(3, min_periods=1).mean().round(2)
        st_df["pm25_ma7"] = st_df["pm25"].rolling(7, min_periods=1).mean().round(2)
        st_df["delta_pm25"] = (st_df["pm25"] - st_df["pm25_previo"]).round(2)
        st_df["pm25_manana"] = st_df["pm25"].shift(-1)
        st_df["categoria_manana"] = st_df["categoria_ica"].shift(-1)
        st_df["color_manana"] = st_df["color_ica"].shift(-1)

        station_frames.append(st_df)

    df_final = pd.concat(station_frames, ignore_index=True)
    dt_col = pd.to_datetime(df_final["fecha"])
    df_final["anio"] = dt_col.dt.year
    df_final["mes"] = dt_col.dt.month
    df_final["dia_semana"] = dt_col.dt.dayofweek
    df_final["periodo_poeca"] = df_final["mes"].isin([2, 3, 4, 10, 11]).map(
        {True: "Gestión Episodios (POECA)", False: "Periodo Regular"}
    )

    df_final = df_final.sort_values(["estacion", "fecha"]).reset_index(drop=True)
    df_final.to_csv(PROCESSED_CSV, index=False)

    manifest = {
        "fuente_primaria": "Portal de Datos Abiertos de Colombia - IDEAM SISAIRE / AMVA (SIATA)",
        "endpoint_horario": SOCRATA_HOURLY_URL,
        "endpoint_anual": SOCRATA_ANNUAL_URL,
        "fecha_extraccion_utc": datetime.utcnow().isoformat() + "Z",
        "rango_temporal": "2024-01-01 a 2025-12-31",
        "total_registros_horarios_crudos": int(len(df_raw)),
        "total_registros_horarios_validos": int(len(df_clean_h)),
        "total_registros_diarios_consolidados": int(len(df_final)),
        "estaciones": list(STATIONS_METADATA.keys()),
        "conteo_mensual_horario": dict(sorted(monthly_counts.items())),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Dataset consolidado guardado en {PROCESSED_CSV}: "
        f"{len(df_final):,} estación-días a partir de {len(df_raw):,} mediciones horarias reales."
    )
    return df_final


def load_air_quality_data() -> pd.DataFrame:
    """
    Carga el dataset consolidado local (`Plan B` activo por defecto).
    Si no existe aún en disco, ejecuta la descarga desde `datos.gov.co`.
    """
    if not PROCESSED_CSV.exists():
        return download_and_build_dataset(force_download=True)
    df = pd.read_csv(PROCESSED_CSV)
    return df


def get_station_summary(df: Optional[pd.DataFrame] = None, fecha: Optional[str] = None) -> pd.DataFrame:
    """
    Devuelve el estado instantáneo de las 5 estaciones para una fecha dada
    (o la última fecha disponible si `fecha=None`).
    """
    if df is None:
        df = load_air_quality_data()
    target_date = fecha if fecha else str(df["fecha"].max())
    sub = df[df["fecha"] == target_date].copy()
    if sub.empty:
        raise ValueError(
            f"No hay registros para la fecha '{target_date}'. "
            f"Rango disponible: {df['fecha'].min()} a {df['fecha'].max()}."
        )
    sub["hex_color"] = sub["color_ica"].map(COLOR_MAP_HEX).fillna("#95a5a6")
    return sub.sort_values("pm25", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df_built = download_and_build_dataset(force_download=True)
    print(df_built.groupby(["estacion", "color_ica"]).size().unstack(fill_value=0))

