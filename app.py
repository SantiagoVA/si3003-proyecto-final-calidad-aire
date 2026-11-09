"""
Dashboard Interactivo de Vigilancia y Asistencia para la Calidad del Aire en el Valle de Aburrá.
Proyecto Final Integrador SI3003 - Inteligencia Artificial | Universidad EAFIT.

Ejecución en un solo comando:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent_core import AirQualityAgent
from src.classical_model import (
    STATE_LABELS,
    load_or_train_classical_model,
    predict_station_air_quality,
)
from src.data_loader import (
    CATEGORY_TO_COLOR,
    COLOR_MAP_HEX,
    STATIONS_METADATA,
    get_station_summary,
    load_air_quality_data,
)
from src.rag_engine import get_rag_engine

st.set_page_config(
    page_title="Vigilancia Calidad del Aire Valle de Aburrá | SIATA - POECA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos limpios y profesionales
st.markdown(
    """
    <style>
    .kpi-card {
        border-radius: 12px;
        padding: 16px 18px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 0.84rem;
        color: #4b5563;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _cached_dataset() -> pd.DataFrame:
    return load_air_quality_data()


@st.cache_resource(show_spinner=False)
def _cached_model_and_rag():
    hmm, artifacts = load_or_train_classical_model()
    rag = get_rag_engine()
    return hmm, artifacts, rag


df = _cached_dataset()
hmm_model, model_artifacts, rag_engine = _cached_model_and_rag()

min_dt = pd.to_datetime(df["fecha"].min()).date()
max_dt = pd.to_datetime(df["fecha"].max()).date()

# ==============================================================================
# SIDEBAR: CONTROLES INTERACTIVOS Y SIMULADOR METEOROLÓGICO
# ==============================================================================
with st.sidebar:
    st.markdown("## 🌿 Control Valle de Aburrá")
    st.caption("Datos Reales IDEAM-SISAIRE / Red SIATA-AMVA (2024–2025)")

    station_list = list(STATIONS_METADATA.keys())
    selected_station = st.selectbox(
        "📍 Estación de Monitoreo",
        options=station_list,
        index=0,
        help="Selecciona la estación oficial del Valle de Aburrá para analizar.",
    )

    preset_mode = st.radio(
        "📅 Escenario Temporal Rápido",
        options=[
            "Episodio Crítico POECA (2024-03-01)",
            "Pico Regional Sur/Centro (2024-03-18)",
            "Cierre Histórico Reciente (2025-12-31)",
            "Selección Manual de Fecha",
        ],
        index=0,
    )
    if preset_mode.startswith("Episodio Crítico"):
        selected_date_obj = pd.to_datetime("2024-03-01").date()
    elif preset_mode.startswith("Pico Regional"):
        selected_date_obj = pd.to_datetime("2024-03-18").date()
    elif preset_mode.startswith("Cierre Histórico"):
        selected_date_obj = max_dt
    else:
        selected_date_obj = st.date_input(
            "Fecha de inspección",
            value=pd.to_datetime("2024-03-01").date(),
            min_value=min_dt,
            max_value=max_dt,
        )

    selected_date = selected_date_obj.strftime("%Y-%m-%d")
    window_days = st.select_slider(
        "📊 Ventana Serie Temporal (días)",
        options=[30, 60, 90, 180, 365],
        value=90,
    )

    # Valores observados en la estación y fecha seleccionada como punto de partida del simulador
    cur_row_df = df[(df["estacion"] == selected_station) & (df["fecha"] == selected_date)]
    if cur_row_df.empty:
        cur_row = df[df["estacion"] == selected_station].iloc[-1]
    else:
        cur_row = cur_row_df.iloc[0]

    st.divider()
    st.markdown("### 🎛️ Simulador Climático (Parte 1)")
    st.caption("Modifica condiciones para predecir mañana ($t+1$) con el modelo propio Gaussian HMM:")

    sim_pm25_prev = st.slider(
        "PM2.5 Previo / Actual (µg/m³)",
        min_value=3.0,
        max_value=110.0,
        value=float(round(cur_row["pm25"], 1)),
        step=0.5,
    )
    sim_wind = st.slider(
        "Velocidad del Viento (m/s)",
        min_value=0.1,
        max_value=5.0,
        value=float(round(cur_row["velocidad_viento"], 2)),
        step=0.1,
    )
    sim_hum = st.slider(
        "Humedad Relativa (%)",
        min_value=25.0,
        max_value=98.0,
        value=float(round(cur_row["humedad_relativa"], 1)),
        step=1.0,
    )
    sim_temp = st.slider(
        "Temperatura Media (°C)",
        min_value=14.0,
        max_value=32.0,
        value=float(round(cur_row["temperatura"], 1)),
        step=0.5,
    )
    sim_rain = st.slider(
        "Precipitación Acumulada (mm)",
        min_value=0.0,
        max_value=40.0,
        value=float(round(cur_row["precipitacion"], 1)),
        step=1.0,
    )

    st.divider()
    with st.expander("⚙️ Configuración del Agente LLM", expanded=False):
        llm_provider_choice = st.selectbox(
            "Backend LLM",
            options=["Auto (Env / Local Qwen2.5 + ReAct)", "Groq (OpenAI Compatible)", "NVIDIA NIM", "OpenAI"],
            index=0,
        )
        user_api_key = st.text_input("API Key Opcional", type="password", value="")
        t_max_cfg = st.slider("Límite de pasos (T_max)", min_value=2, max_value=8, value=6)

# ==============================================================================
# CÁLCULOS REACTIVOS PARA LA VISTA ACTUAL
# ==============================================================================
valley_snapshot = get_station_summary(df, fecha=selected_date)
sim_prediction = predict_station_air_quality(
    estacion=selected_station,
    viento=sim_wind,
    humedad=sim_hum,
    pm25_previo=sim_pm25_prev,
    temperatura=sim_temp,
    precipitacion=sim_rain,
    es_periodo_poeca=1 if selected_date_obj.month in (2, 3, 4, 10, 11) else 0,
)

# Evaluar proporción de estaciones poblacionales sobre 37 ug/m3 en la fecha seleccionada
pop_stations = valley_snapshot[valley_snapshot["estacion"] != "Centro"]
pct_orange_plus = 100.0 * float((pop_stations["pm25"] > 37.0).mean())
pct_red_plus = 100.0 * float((pop_stations["pm25"] >= 55.5).mean())
if pct_red_plus >= 50.0:
    valley_poeca_status = "🔴 ALERTA POECA (Nivel II)"
elif pct_orange_plus >= 50.0:
    valley_poeca_status = "🟠 PREVENCIÓN POECA (Nivel I)"
elif (valley_snapshot["pm25"] > 37.0).any():
    valley_poeca_status = "🟡 VIGILANCIA FOCALIZADA"
else:
    valley_poeca_status = "🟢 NIVEL NORMAL"

# ==============================================================================
# ENCABEZADO Y TARJETAS KPI
# ==============================================================================
st.title("🌄 Sistema Inteligente de Vigilancia de Calidad del Aire — Valle de Aburrá")
st.markdown(
    f"**Universidad EAFIT — SI3003 Inteligencia Artificial** | Estación activa: "
    f"**{cur_row['estacion_oficial']}** | Fecha: **`{selected_date}`** (*{cur_row['periodo_poeca']}*)"
)

obs_hex = COLOR_MAP_HEX.get(str(cur_row["color_ica"]), "#6b7280")
pred_hex = COLOR_MAP_HEX.get(sim_prediction["color_ica_predicho"], "#6b7280")

k1, k2, k3, k4 = st.columns(4)
with k1:
    delta_norma = float(cur_row["pm25"]) - 37.0
    sign_str = f"+{delta_norma:.1f}" if delta_norma > 0 else f"{delta_norma:.1f}"
    st.markdown(
        f"""<div class="kpi-card" style="border-left: 6px solid {obs_hex};">
        <div class="kpi-label">PM2.5 Promedio 24h ({selected_station})</div>
        <div class="kpi-value">{cur_row['pm25']:.1f} µg/m³</div>
        <div class="kpi-sub">{sign_str} µg/m³ frente a norma diaria (37.0 µg/m³)</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card" style="border-left: 6px solid {obs_hex};">
        <div class="kpi-label">Índice ICA Oficial (Res. 2254)</div>
        <div class="kpi-value">{cur_row['ica']:.0f} — {cur_row['color_ica']}</div>
        <div class="kpi-sub">{cur_row['categoria_ica']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card" style="border-left: 6px solid {pred_hex};">
        <div class="kpi-label">Pronóstico Mañana (Gaussian HMM)</div>
        <div class="kpi-value">{sim_prediction['color_ica_predicho']} ({sim_prediction['confianza']*100:.0f}%)</div>
        <div class="kpi-sub">{sim_prediction['categoria_predicha']} (~{sim_prediction['pm25_estimado_ugm3']} µg/m³)</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card" style="border-left: 6px solid #2563eb;">
        <div class="kpi-label">Estado Regional Protocolo POECA</div>
        <div class="kpi-value" style="font-size:1.25rem;">{valley_poeca_status}</div>
        <div class="kpi-sub">Viento: {cur_row['velocidad_viento']:.2f} m/s | Humedad: {cur_row['humedad_relativa']:.0f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ==============================================================================
# SECCIÓN PRINCIPAL DE PESTAÑAS: MONITOREO + MODELO CLÁSICO + CHAT AGENTE
# ==============================================================================
tab_dash, tab_chat, tab_model = st.tabs(
    [
        "🗺️ Mapa Geoespacial y Serie Temporal POECA",
        "🤖 Chat con Agente Inteligente (RAG + Tools)",
        "📐 Auditoría del Modelo Clásico (Parte 1 - NumPy)",
    ]
)

with tab_dash:
    col_map, col_ts = st.columns([1.0, 1.35], gap="medium")

    with col_map:
        st.subheader(f"Mapa de Estaciones SIATA/AMVA ({selected_date})")
        marker_sizes = np.clip(valley_snapshot["pm25"].to_numpy() * 0.65 + 14.0, 16.0, 42.0)
        hover_texts = [
            (
                f"<b>{r['estacion']}</b> ({r['municipio']})<br>"
                f"Estación: {r['estacion_oficial']}<br>"
                f"PM2.5 (24h): <b>{r['pm25']:.1f} µg/m³</b> (Máx horario: {r['pm25_max_horario']:.1f})<br>"
                f"ICA: <b>{r['ica']:.1f}</b> — {r['categoria_ica']} ({r['color_ica']})<br>"
                f"Viento: {r['velocidad_viento']:.2f} m/s | Humedad: {r['humedad_relativa']:.1f}% | Temp: {r['temperatura']:.1f} °C"
            )
            for _, r in valley_snapshot.iterrows()
        ]

        fig_map = go.Figure()
        # Contorno exterior oscuro para resaltar los pines semafóricos sobre OpenStreetMap
        fig_map.add_trace(
            go.Scattermap(
                lat=valley_snapshot["latitud"],
                lon=valley_snapshot["longitud"],
                mode="markers",
                marker=dict(size=marker_sizes + 5, color="#1f2937", opacity=0.85),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig_map.add_trace(
            go.Scattermap(
                lat=valley_snapshot["latitud"],
                lon=valley_snapshot["longitud"],
                mode="markers+text",
                text=[f"{r['estacion']} ({r['pm25']:.0f})" for _, r in valley_snapshot.iterrows()],
                textposition="top right",
                marker=dict(
                    size=marker_sizes,
                    color=valley_snapshot["hex_color"],
                    opacity=0.95,
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Estaciones AMVA",
            )
        )
        fig_map.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=6.252, lon=-75.572),
                zoom=10.5,
            ),
            margin=dict(l=0, r=0, t=8, b=0),
            height=440,
            showlegend=False,
        )
        st.plotly_chart(fig_map, width="stretch")

        st.dataframe(
            valley_snapshot[
                ["estacion", "municipio", "pm25", "ica", "color_ica", "velocidad_viento", "humedad_relativa"]
            ].rename(
                columns={
                    "estacion": "Estación",
                    "municipio": "Municipio",
                    "pm25": "PM2.5 (µg/m³)",
                    "ica": "ICA",
                    "color_ica": "Color ICA",
                    "velocidad_viento": "Viento (m/s)",
                    "humedad_relativa": "Humedad (%)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with col_ts:
        st.subheader(f"Evolución Temporal de PM2.5 y Umbrales POECA — {selected_station}")
        half_w = timedelta(days=int(window_days // 2))
        start_w = max(min_dt, selected_date_obj - half_w).strftime("%Y-%m-%d")
        end_w = min(max_dt, selected_date_obj + half_w).strftime("%Y-%m-%d")

        sub_ts = df[
            (df["estacion"] == selected_station)
            & (df["fecha"] >= start_w)
            & (df["fecha"] <= end_w)
        ].sort_values("fecha")

        fig_ts = go.Figure()
        # Bandas de fondo normativas Resolución 2254 / POECA
        fig_ts.add_hrect(y0=0, y1=12.0, fillcolor="#2ecc71", opacity=0.08, line_width=0)
        fig_ts.add_hrect(y0=12.0, y1=37.0, fillcolor="#f1c40f", opacity=0.08, line_width=0)
        fig_ts.add_hrect(y0=37.0, y1=55.4, fillcolor="#e67e22", opacity=0.10, line_width=0)
        fig_ts.add_hrect(y0=55.4, y1=max(80.0, float(sub_ts["pm25"].max()) + 8.0), fillcolor="#e74c3c", opacity=0.10, line_width=0)

        fig_ts.add_trace(
            go.Scatter(
                x=sub_ts["fecha"],
                y=sub_ts["pm25"],
                mode="lines+markers",
                name="PM2.5 Diario (24h)",
                line=dict(color="#1f2937", width=1.8),
                marker=dict(
                    size=6,
                    color=[COLOR_MAP_HEX.get(c, "#6b7280") for c in sub_ts["color_ica"]],
                    line=dict(width=0.8, color="#111827"),
                ),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=sub_ts["fecha"],
                y=sub_ts["pm25_ma7"],
                mode="lines",
                name="Media Móvil 7 Días (MA-7)",
                line=dict(color="#2563eb", width=2.6),
            )
        )

        # Líneas horizontales normativas POECA
        fig_ts.add_hline(
            y=12.0,
            line_dash="dot",
            line_color="#16a34a",
            annotation_text="Corte Buena/Aceptable (12 µg/m³)",
            annotation_position="bottom right",
        )
        fig_ts.add_hline(
            y=37.0,
            line_dash="dash",
            line_color="#d97706",
            annotation_text="Prevención POECA / Norma Nacional (37 µg/m³)",
            annotation_position="top right",
        )
        fig_ts.add_hline(
            y=55.5,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Alerta Roja POECA (55.5 µg/m³)",
            annotation_position="top right",
        )
        fig_ts.add_vline(x=selected_date, line_width=2, line_dash="solid", line_color="#7c3aed")

        fig_ts.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Concentración PM2.5 (µg/m³)",
            height=400,
            margin=dict(l=10, r=10, t=12, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, width="stretch")

        # Distribución de probabilidad posterior del simulador en vivo
        st.markdown("#### 🎯 Distribución Posterior Predicha para Mañana ($t+1$) — Simulador Activo")
        prob_labels = list(sim_prediction["probabilidades_por_estado"].keys())
        prob_vals = [v * 100.0 for v in sim_prediction["probabilidades_por_estado"].values()]
        bar_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

        fig_prob = go.Figure(
            go.Bar(
                x=prob_vals,
                y=prob_labels,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{p:.1f}%" for p in prob_vals],
                textposition="outside",
            )
        )
        fig_prob.update_layout(
            xaxis=dict(title="Probabilidad Posterior (%)", range=[0, 108]),
            yaxis=dict(autorange="reversed"),
            height=210,
            margin=dict(l=10, r=30, t=6, b=10),
        )
        st.plotly_chart(fig_prob, width="stretch")

# ==============================================================================
# TAB 2: INTERFAZ CONVERSACIONAL CON EL AGENTE INTELIGENTE
# ==============================================================================
with tab_chat:
    st.subheader("🤖 Asistente Técnico Conversacional SIATA / POECA")
    st.caption(
        "Integra consultas de datos reales (`get_air_quality_data`), modelo probabilístico propio "
        "(`predict_air_quality`) y recuperación normativa FAISS (`search_poeca_protocols`) con loop `T_max`."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "¡Hola! Soy el **Agente Inteligente de Calidad del Aire del Valle de Aburrá**. "
                    "Puedo consultar las mediciones históricas y actuales de las estaciones del SIATA/AMVA, "
                    "pronosticar el estado ICA del día siguiente con nuestro modelo probabilístico **Gaussian HMM** "
                    "y explicarte las restricciones del **Protocolo POECA** y la **Resolución 2254 de 2017** con citas verificables."
                ),
                "trace": None,
            }
        ]

    # Botones de preguntas rápidas para demostración fluida
    st.markdown("**💡 Consultas sugeridas de un clic:**")
    q_cols = st.columns(4)
    quick_prompts = [
        f"¿Cómo estuvo la estación {selected_station} en la fecha {selected_date} y qué medidas del POECA aplican?",
        f"Predice mañana en {selected_station} con viento {sim_wind} m/s, humedad {sim_hum}% y PM2.5 previo {sim_pm25_prev} ug/m3.",
        "¿Qué restricciones de Pico y Placa y camiones aplican en Estado de Alerta Roja del POECA?",
        "¿Quiénes son los grupos sensibles y qué mascarilla sirve para protegerse del PM2.5?",
    ]
    triggered_prompt = None
    labels_btn = [
        f"📍 Analizar {selected_station} ({selected_date})",
        "🎯 Ejecutar Pronóstico Simulador",
        "📜 Reglas Alerta Roja POECA",
        "🩺 Salud y Grupos Sensibles",
    ]
    for col_b, lbl, qp in zip(q_cols, labels_btn, quick_prompts):
        if col_b.button(lbl, width="stretch"):
            triggered_prompt = qp

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                tr = msg["trace"]
                with st.expander(
                    f"🔍 Visor de Auditoría del Loop Agéntico — Pasos: {tr['total_steps']} | "
                    f"Tools: {', '.join(tr['tools_called']) or 'Ninguna'} | Backend: {tr['backend_used']}"
                ):
                    for s in tr["steps"]:
                        st.markdown(f"**Paso {s['step']}** ({s['latency_ms']} ms) — *Thought:* {s['thought']}")
                        if s.get("action"):
                            st.code(
                                f"Action: {s['action']}({json.dumps(s['action_input'], ensure_ascii=False)})",
                                language="python",
                            )
                        if s.get("observation"):
                            obs_preview = s["observation"]
                            if len(obs_preview) > 900:
                                obs_preview = obs_preview[:900] + " ... [truncado]"
                            if s.get("is_error"):
                                st.error(obs_preview)
                            else:
                                st.json(json.loads(s["observation"]) if s["observation"].startswith("{") else {"obs": obs_preview})

    user_input = st.chat_input("Escribe tu pregunta sobre PM2.5, pronósticos o normativa POECA...")
    active_query = triggered_prompt or user_input

    if active_query:
        st.session_state.chat_history.append({"role": "user", "content": active_query, "trace": None})
        with st.chat_message("user"):
            st.markdown(active_query)

        prov_map = {
            "Auto (Env / Local Qwen2.5 + ReAct)": None,
            "Groq (OpenAI Compatible)": "groq",
            "NVIDIA NIM": "nvidia",
            "OpenAI": "openai",
        }
        agent = AirQualityAgent(
            t_max=t_max_cfg,
            api_key=user_api_key if user_api_key else None,
            provider=prov_map[llm_provider_choice],
        )
        with st.chat_message("assistant"):
            with st.spinner("Ejecutando loop agéntico, herramientas y verificación documental..."):
                resp = agent.ask(active_query)
            st.markdown(resp.answer)
            trace_dict = resp.to_dict()
            with st.expander(
                f"🔍 Visor de Auditoría del Loop Agéntico — Pasos: {resp.total_steps} | "
                f"Tools: {', '.join(resp.tools_called) or 'Ninguna'} | Backend: {resp.backend_used}"
            ):
                for s in trace_dict["steps"]:
                    st.markdown(f"**Paso {s['step']}** ({s['latency_ms']} ms) — *Thought:* {s['thought']}")
                    if s.get("action"):
                        st.code(
                            f"Action: {s['action']}({json.dumps(s['action_input'], ensure_ascii=False)})",
                            language="python",
                        )
                    if s.get("observation"):
                        if s.get("is_error"):
                            st.error(s["observation"])
                        else:
                            st.json(json.loads(s["observation"]) if s["observation"].startswith("{") else {"obs": s["observation"]})

        st.session_state.chat_history.append(
            {"role": "assistant", "content": resp.answer, "trace": trace_dict}
        )

# ==============================================================================
# TAB 3: AUDITORÍA DEL MODELO CLÁSICO DE LA PARTE 1
# ==============================================================================
with tab_model:
    st.subheader("📐 Evaluación Experimental de la Parte 1 (Implementación Propia en NumPy)")
    st.markdown(
        f"**Partición Temporal Estricta:** Entrenamiento en **2024** (`n = {model_artifacts['n_train']:,}` estación-días) "
        f"→ Evaluación Hold-out en **2025** (`n = {model_artifacts['n_test']:,}` estación-días)."
    )

    comp_rows = []
    for m_name, m_vals in model_artifacts["models"].items():
        comp_rows.append(
            {
                "Modelo Evaluado": m_name,
                "Accuracy": m_vals["accuracy"],
                "Macro F1-Score": m_vals["macro_f1"],
                "Weighted F1-Score": m_vals["weighted_f1"],
                "Brier Score (↓ mejor)": m_vals["brier_score"],
            }
        )
    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("#### Matriz de Transición Aprendida $A_{ij} = P(S_{t+1}=j \\mid S_t=i)$")
        trans_mat = np.array(model_artifacts["hmm_params"]["transmat"])
        short_lbls = ["Verde (Buena)", "Amarillo (Acept.)", "Naranja (Sensib.)", "Rojo (Dañina)"]
        fig_tr = go.Figure(
            go.Heatmap(
                z=np.round(trans_mat, 3),
                x=short_lbls,
                y=short_lbls,
                colorscale="Blues",
                text=np.round(trans_mat, 3),
                texttemplate="%{text}",
            )
        )
        fig_tr.update_layout(
            xaxis_title="Estado Día Siguiente (t+1)",
            yaxis_title="Estado Actual (t)",
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_tr, width="stretch")

    with c_m2:
        st.markdown("#### Matriz de Confusión en Hold-out 2025 (Gaussian HMM Propio)")
        cm_hmm = np.array(
            model_artifacts["models"]["Gaussian HMM + Filtro Bayesiano (Propio - NumPy)"]["confusion_matrix"]
        )
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm_hmm,
                x=short_lbls,
                y=short_lbls,
                colorscale="Greens",
                text=cm_hmm,
                texttemplate="%{text}",
            )
        )
        fig_cm.update_layout(
            xaxis_title="Categoría Predicha",
            yaxis_title="Categoría Real Observada",
            yaxis=dict(autorange="reversed"),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, width="stretch")
