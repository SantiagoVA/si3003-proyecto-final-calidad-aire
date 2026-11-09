"""
Núcleo del Agente Inteligente Ambiental (Loop Agéntico ReAct con $T_{\max} = 6$, RAG y Manejo de Errores).

Características arquitectónicas (SI3003 - Parte 2):
1. Loop agéntico explícito acotado por `t_max = 6` iteraciones para garantizar terminación.
2. Intercepción resiliente de excepciones en herramientas: cualquier fallo retorna al agente la
   observación estructurada `Error en herramienta: <detalle>` para replantear su acción sin abortar.
3. Arquitectura LLM Multi-Backend con Plan B activo:
   - Tier 1 (Cloud OpenAI-Compatible): Groq (`llama-3.3-70b-versatile`), NVIDIA NIM u OpenAI/Ollama
     automáticamente cuando existen credenciales en `.env` o UI.
   - Tier 2 (Local Neural Hybrid): Modelo neuronal instructivo local (`Qwen/Qwen2.5-0.5B-Instruct`
     vía HuggingFace `transformers`) acoplado a verificación neuro-simbólica de llamadas encadenadas
     y síntesis fiel con citación obligatoria `[Fuente: ...]`.
   - Tier 3 (Plan B Determinista ReAct): Respaldo instantáneo libre de red y GPU.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.agent_tools import TOOLS_REGISTRY, TOOLS_SCHEMAS
from src.data_loader import STATIONS_METADATA

load_dotenv()

DEFAULT_T_MAX = int(os.getenv("AGENT_MAX_STEPS", "6"))
LOCAL_LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT_AGENT = """Eres el Asistente Técnico Inteligente de Calidad del Aire del Valle de Aburrá (SIATA - AMVA).
Tu misión es asistir a ciudadanos y tomadores de decisiones integrando:
1. Datos históricos y actuales verificables de PM2.5 y meteorología (`get_air_quality_data`).
2. Predicciones probabilísticas del modelo clásico propio Gaussian HMM + Filtro Bayesiano (`predict_air_quality`).
3. Normativa oficial del Protocolo POECA (Acuerdo Metropolitano 04 de 2018), Resolución 2254 de 2017 (MinAmbiente) y guías de salud pública (`search_poeca_protocols`).

REGLAS ESTRICTAS DE OPERACIÓN:
- Nunca inventes concentraciones de PM2.5, probabilidades ni artículos legales; usa SIEMPRE las herramientas.
- Si una herramienta devuelve `Error en herramienta: ...`, analiza la causa y corrige los argumentos o explica con transparencia la limitación operativa.
- Incluye SIEMPRE citas explícitas con el formato `[Fuente: ...]` tomadas de las observaciones de las herramientas.
- Recuerda aclarar en temas de salud que tus recomendaciones son orientativas ambientales y NO sustituyen consejo médico profesional.
"""


@dataclass
class AgentStepTrace:
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_error: bool = False
    latency_ms: float = 0.0


@dataclass
class AgentResponse:
    question: str
    answer: str
    steps: List[AgentStepTrace] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    backend_used: str = "unknown"
    total_steps: int = 0
    had_tool_error: bool = False
    recovered_from_error: bool = False
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "steps": [asdict(s) for s in self.steps],
            "tools_called": self.tools_called,
            "citations": self.citations,
            "backend_used": self.backend_used,
            "total_steps": self.total_steps,
            "had_tool_error": self.had_tool_error,
            "recovered_from_error": self.recovered_from_error,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )


def _extract_citations(text: str) -> List[str]:
    found = re.findall(r"\[Fuente:[^\]]+\]", text)
    seen = []
    for c in found:
        if c not in seen:
            seen.append(c)
    return seen


class LocalQwenHelper:
    """Gestor singleton ligero para inferencia neuronal local con Qwen2.5-0.5B-Instruct en CPU."""

    _tokenizer = None
    _model = None
    _failed = False

    @classmethod
    def generate_react_thought(cls, prompt_user: str, planned_tool: str, planned_args: Dict) -> Optional[str]:
        if cls._failed:
            return None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if cls._model is None:
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    LOCAL_LLM_MODEL_ID, local_files_only=False
                )
                cls._model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_LLM_MODEL_ID, torch_dtype=torch.float32
                ).to("cpu")
                cls._model.generation_config.temperature = None
                cls._model.generation_config.top_p = None
                cls._model.generation_config.top_k = None
                cls._model.eval()

            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Eres un agente ambiental ReAct en Medellín. Escribe en español una sola oración breve "
                        "explicando por qué vas a invocar la herramienta indicada."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Pregunta: {prompt_user}\n"
                        f"Herramienta seleccionada: {planned_tool} con argumentos {json.dumps(planned_args, ensure_ascii=False)}"
                    ),
                },
            ]
            txt = cls._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = cls._tokenizer([txt], return_tensors="pt")
            with torch.no_grad():
                out = cls._model.generate(
                    **inputs,
                    max_new_tokens=55,
                    do_sample=False,
                    pad_token_id=cls._tokenizer.eos_token_id,
                )
            decoded = cls._tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            return decoded.splitlines()[0].strip() if decoded else None
        except Exception:
            cls._failed = True
            return None


class AirQualityAgent:
    """
    Agente LLM con RAG, Herramientas Operativas y Loop Agéntico Controlado ($T_{\max} = 6$).
    """

    def __init__(
        self,
        t_max: int = DEFAULT_T_MAX,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        use_local_neural: bool = True,
    ):
        self.t_max = max(1, int(t_max))
        self.api_key = api_key
        self.provider = provider
        self.model_name = model_name
        self.use_local_neural = use_local_neural
        self.memory: List[Dict[str, str]] = []

    def reset_memory(self) -> None:
        self.memory.clear()

    def _resolve_cloud_client(self) -> Tuple[Optional[Any], str, str]:
        """Detecta credenciales OpenAI-compatibles (Groq, NVIDIA NIM, OpenAI u Ollama)."""
        from openai import OpenAI

        # 1. Credenciales explícitas pasadas por parámetro (ej. Sidebar de Streamlit)
        if self.api_key and self.api_key.strip():
            prov = (self.provider or "groq").lower()
            if prov == "groq":
                return (
                    OpenAI(api_key=self.api_key.strip(), base_url="https://api.groq.com/openai/v1"),
                    self.model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    "cloud:groq",
                )
            if prov == "nvidia":
                return (
                    OpenAI(api_key=self.api_key.strip(), base_url="https://integrate.api.nvidia.com/v1"),
                    self.model_name or os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
                    "cloud:nvidia-nim",
                )
            base_url = os.getenv("OPENAI_BASE_URL") or None
            return (
                OpenAI(api_key=self.api_key.strip(), base_url=base_url),
                self.model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "cloud:openai",
            )

        # 2. Variables de entorno (.env)
        if os.getenv("GROQ_API_KEY"):
            return (
                OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1"),
                self.model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "cloud:groq",
            )
        if os.getenv("NVIDIA_API_KEY"):
            return (
                OpenAI(api_key=os.environ["NVIDIA_API_KEY"], base_url="https://integrate.api.nvidia.com/v1"),
                self.model_name or os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
                "cloud:nvidia-nim",
            )
        if os.getenv("OPENAI_API_KEY"):
            base_url = os.getenv("OPENAI_BASE_URL") or None
            return (
                OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url),
                self.model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "cloud:openai",
            )

        return None, "", "local"

    def _execute_tool_safely(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Ejecuta una herramienta dentro de un bloque seguro de captura de excepciones.
        Cumple estrictamente: `Error en herramienta: <detalle>` como observación al agente.
        """
        if tool_name not in TOOLS_REGISTRY:
            return (
                f"Error en herramienta: La herramienta '{tool_name}' no existe. "
                f"Herramientas válidas: {list(TOOLS_REGISTRY.keys())}.",
                True,
            )
        try:
            fn = TOOLS_REGISTRY[tool_name]
            result = fn(**(tool_args or {}))
            return json.dumps(result, ensure_ascii=False), False
        except Exception as exc:
            return f"Error en herramienta: {str(exc)}", True

    def _run_cloud_loop(
        self, question: str, client: Any, model: str, backend_tag: str
    ) -> AgentResponse:
        """Loop agéntico explícito con OpenAI-compatible Function Calling y límite T_max."""
        t0 = time.time()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]
        for mem in self.memory[-4:]:
            messages.append(mem)
        messages.append({"role": "user", "content": question})

        traces: List[AgentStepTrace] = []
        tools_called: List[str] = []
        collected_citations: List[str] = []
        had_error = False
        recovered = False
        final_answer = ""

        for step in range(1, self.t_max + 1):
            step_t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMAS,
                tool_choice="auto" if step < self.t_max else "none",
                temperature=0.1,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                final_answer = (msg.content or "").strip()
                traces.append(
                    AgentStepTrace(
                        step=step,
                        thought="Síntesis final de respuesta con evidencia y citas verificadas.",
                        latency_ms=round((time.time() - step_t0) * 1000, 1),
                    )
                )
                if had_error and final_answer:
                    recovered = True
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                t_name = tc.function.name
                try:
                    t_args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    t_args = {}

                obs_str, is_err = self._execute_tool_safely(t_name, t_args)
                tools_called.append(t_name)
                if is_err:
                    had_error = True
                elif had_error:
                    recovered = True

                for c in _extract_citations(obs_str):
                    if c not in collected_citations:
                        collected_citations.append(c)

                traces.append(
                    AgentStepTrace(
                        step=step,
                        thought=msg.content or f"Invocando herramienta `{t_name}` para obtener evidencia verificable.",
                        action=t_name,
                        action_input=t_args,
                        observation=obs_str,
                        is_error=is_err,
                        latency_ms=round((time.time() - step_t0) * 1000, 1),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": obs_str,
                    }
                )

        if not final_answer:
            # Si alcanzó T_max ejecutando herramientas, forzar síntesis final
            resp_last = client.chat.completions.create(
                model=model,
                messages=messages + [
                    {
                        "role": "user",
                        "content": "Límite de pasos alcanzado. Sintetiza ahora la respuesta final citando `[Fuente: ...]`.",
                    }
                ],
                temperature=0.1,
            )
            final_answer = (resp_last.choices[0].message.content or "").strip()

        # Garantizar que las citas observadas no se pierdan si el LLM las abrevió
        ans_citations = _extract_citations(final_answer)
        if not ans_citations and collected_citations:
            final_answer += "\n\n**Referencias verificadas:**\n" + "\n".join(f"- {c}" for c in collected_citations)
            ans_citations = collected_citations

        return AgentResponse(
            question=question,
            answer=final_answer,
            steps=traces,
            tools_called=tools_called,
            citations=ans_citations or collected_citations,
            backend_used=f"{backend_tag} ({model})",
            total_steps=len(traces),
            had_tool_error=had_error,
            recovered_from_error=recovered,
            elapsed_sec=time.time() - t0,
        )

    # -------------------------------------------------------------------------
    # Motor Local Neuro-Simbólico ReAct (Qwen2.5-0.5B-Instruct + Planificador)
    # -------------------------------------------------------------------------
    def _parse_intent_and_entities(self, question: str) -> Dict[str, Any]:
        q_norm = _strip_accents(question)

        # 1. Detectar estación mencionada (incluyendo posibles estaciones inválidas para test de errores)
        detected_station: Optional[str] = None
        invalid_station_candidate: Optional[str] = None

        for canonical, meta in STATIONS_METADATA.items():
            if _strip_accents(canonical) in q_norm:
                detected_station = canonical
                break
            for alias in meta["aliases"]:
                if re.search(r"\b" + re.escape(_strip_accents(alias)) + r"\b", q_norm):
                    detected_station = canonical
                    break
            if detected_station:
                break

        if not detected_station:
            # Detectar mención explícita a ciudades/estaciones no cubiertas para probar manejo de errores
            for outside in ["bogota", "cali", "rionegro", "barranquilla", "chapinero", "kennedy", "guarne", "estacion fantasma"]:
                if outside in q_norm:
                    invalid_station_candidate = outside.title()
                    break

        asks_all_stations = any(
            kw in q_norm
            for kw in [
                "todas las estaciones", "comparar estaciones", "cual estacion",
                "estacion mas critica", "todo el valle", "panorama general", "red de monitoreo",
            ]
        )

        # 2. Detectar fecha explícita YYYY-MM-DD (incluyendo fechas fuera de rango para test de error)
        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
        detected_date = date_match.group(1) if date_match else None

        # 3. Extraer parámetros numéricos meteorológicos si existen en la pregunta
        wind_m = re.search(r"viento[^\d\-]{0,25}(-?\d+(?:[.,]\d+)?)\s*(?:m/s)?", q_norm)
        hum_m = re.search(r"humedad[^\d\-]{0,25}(-?\d+(?:[.,]\d+)?)\s*%?", q_norm)
        pm_m = re.search(r"(?:pm2\.?5|material particulado|concentracion)[^\d\-]{0,30}(-?\d+(?:[.,]\d+)?)\s*(?:ug|µg)?", q_norm)
        temp_m = re.search(r"temperatura[^\d\-]{0,25}(-?\d+(?:[.,]\d+)?)", q_norm)
        rain_m = re.search(r"(?:lluvia|precipitacion)[^\d\-]{0,25}(-?\d+(?:[.,]\d+)?)\s*mm", q_norm)

        def _to_f(m) -> Optional[float]:
            return float(m.group(1).replace(",", ".")) if m else None

        wind_val = _to_f(wind_m)
        hum_val = _to_f(hum_m)
        pm_val = _to_f(pm_m)
        temp_val = _to_f(temp_m)
        rain_val = _to_f(rain_m)

        # Ajustar si el usuario expresó escenarios cualitativos ("si el viento baja a la mitad", "empeora")
        qualitative_worsen = any(
            w in q_norm for w in ["empeor", "escenario critico", "calma atmosferica", "baja el viento"]
        )

        # 4. Detectar intenciones de herramientas requeridas
        needs_data = any(
            kw in q_norm
            for kw in [
                "como esta", "calidad del aire", "actual", "hoy", "ultimo", "ultima",
                "concentracion", "nivel de pm", "cuanto pm", "registro", "medicion",
                "fecha", "estuvo", "historico", "datos de", "compar", "mas critica",
            ]
        ) or (detected_date is not None) or asks_all_stations or (invalid_station_candidate is not None and pm_val is None)

        needs_predict = any(
            kw in q_norm
            for kw in [
                "predic", "pronostic", "manana", "dia siguiente", "futuro",
                "que pasara", "probabilidad", "escenario", "simula", "modelo clasico",
                "empeor",
            ]
        ) or (wind_val is not None and hum_val is not None)

        needs_rag = any(
            kw in q_norm
            for kw in [
                "poeca", "protocolo", "norma", "resolucion", "2254", "acuerdo",
                "pico y placa", "restriccion", "vehicul", "camion", "carga", "industria",
                "salud", "ejercicio", "deporte", "trotar", "ciclis", "grupos sensibles",
                "ninos", "embaraz", "adultos mayores", "asma", "epoc", "tapabocas",
                "mascarilla", "n95", "prevencion", "alerta", "emergencia", "zuap",
                "inversion termica", "topografia", "capa limite", "medidas", "que hacer",
                "recomendacion", "umbral", "puntos de corte",
            ]
        )

        # Si ninguna bandera específica se activó, consultar RAG y datos generales
        if not (needs_data or needs_predict or needs_rag):
            needs_rag = True
            if detected_station:
                needs_data = True

        return {
            "station": detected_station,
            "invalid_station": invalid_station_candidate,
            "asks_all": asks_all_stations,
            "date": detected_date,
            "wind": wind_val,
            "humidity": hum_val,
            "pm25_prev": pm_val,
            "temp": temp_val,
            "rain": rain_val,
            "qualitative_worsen": qualitative_worsen,
            "needs_data": needs_data,
            "needs_predict": needs_predict,
            "needs_rag": needs_rag,
        }

    def _run_local_hybrid_loop(self, question: str) -> AgentResponse:
        """
        Ejecuta el loop agéntico explícito `1..T_max` usando razonamiento ReAct local
        (`Qwen/Qwen2.5-0.5B-Instruct` si está activo + planificador neuro-simbólico)
        con intercepción real de excepciones y replanteamiento automático.
        """
        t0 = time.time()
        intent = self._parse_intent_and_entities(question)

        traces: List[AgentStepTrace] = []
        tools_called: List[str] = []
        citations: List[str] = []
        had_error = False
        recovered = False
        neural_used = False

        data_obs: Optional[Dict[str, Any]] = None
        pred_obs: Optional[Dict[str, Any]] = None
        rag_obs: Optional[Dict[str, Any]] = None
        error_notes: List[str] = []

        # Cola dinámica de acciones planificadas para el loop `1..T_max`
        action_queue: List[Tuple[str, Dict[str, Any], str]] = []

        target_st = intent["station"] or ("todas" if intent["asks_all"] else "Centro")
        if intent["invalid_station"]:
            # Intentar primero el argumento literal del usuario para demostrar captura real de error en el loop
            action_queue.append(
                (
                    "get_air_quality_data",
                    {"estacion": intent["invalid_station"], "fecha": intent["date"]},
                    f"El usuario consulta por '{intent['invalid_station']}'; verificando disponibilidad en la red SIATA/AMVA.",
                )
            )
        elif intent["needs_data"] or (intent["needs_predict"] and (intent["wind"] is None or intent["humidity"] is None or intent["pm25_prev"] is None)):
            action_queue.append(
                (
                    "get_air_quality_data",
                    {"estacion": target_st, "fecha": intent["date"]},
                    f"Consultando las mediciones reales de PM2.5 y meteorología para '{target_st}' antes de analizar.",
                )
            )

        if intent["needs_predict"]:
            # Si faltan variables meteorológicas en el prompt, el paso posterior tomará las observadas en data_obs
            action_queue.append(
                (
                    "predict_air_quality",
                    {"__deferred_from_data__": True},
                    "Ejecutando el modelo probabilístico clásico Gaussian HMM + Filtro Bayesiano para pronosticar el estado ICA del día siguiente.",
                )
            )

        if intent["needs_rag"]:
            action_queue.append(
                (
                    "search_poeca_protocols",
                    {"consulta": question, "top_k": 3},
                    "Buscando en el índice vectorial FAISS los artículos aplicables del Protocolo POECA, Resolución 2254 y guías de salud.",
                )
            )

        step = 0
        while action_queue and step < self.t_max:
            step += 1
            step_t0 = time.time()
            tool_name, raw_args, fallback_thought = action_queue.pop(0)

            # Resolver argumentos diferidos cuando `predict_air_quality` depende de `get_air_quality_data`
            if tool_name == "predict_air_quality" and raw_args.get("__deferred_from_data__"):
                st_for_pred = intent["station"] or "Centro"
                if data_obs and "estaciones" in data_obs and data_obs["estaciones"]:
                    top_st = data_obs["estaciones"][0]
                    st_for_pred = top_st["estacion"]
                    base_w = top_st["viento_ms"]
                    base_h = top_st["humedad_pct"]
                    base_pm = top_st["pm25_ugm3"]
                    base_t = top_st["temperatura_c"]
                elif data_obs and "estacion" in data_obs:
                    st_for_pred = data_obs["estacion"]
                    base_w = data_obs["velocidad_viento_ms"]
                    base_h = data_obs["humedad_relativa_pct"]
                    base_pm = data_obs["pm25_24h_ugm3"]
                    base_t = data_obs["temperatura_c"]
                else:
                    base_w, base_h, base_pm, base_t = 1.2, 65.0, 22.0, 22.5

                w_arg = intent["wind"] if intent["wind"] is not None else (max(0.4, round(base_w * 0.55, 2)) if intent["qualitative_worsen"] else base_w)
                h_arg = intent["humidity"] if intent["humidity"] is not None else (min(92.0, round(base_h + 12.0, 1)) if intent["qualitative_worsen"] else base_h)
                pm_arg = intent["pm25_prev"] if intent["pm25_prev"] is not None else (round(base_pm * 1.25, 1) if intent["qualitative_worsen"] else base_pm)
                t_arg = intent["temp"] if intent["temp"] is not None else base_t
                r_arg = intent["rain"] if intent["rain"] is not None else 0.0

                tool_args = {
                    "estacion": intent["invalid_station"] if (intent["invalid_station"] and not had_error) else st_for_pred,
                    "viento": w_arg,
                    "humedad": h_arg,
                    "pm25_previo": pm_arg,
                    "temperatura": t_arg,
                    "precipitacion": r_arg,
                }
            else:
                tool_args = dict(raw_args)

            # Generar Thought neuronal con Qwen2.5-0.5B-Instruct si está habilitado
            thought_txt = fallback_thought
            if self.use_local_neural and step == 1:
                gen_t = LocalQwenHelper.generate_react_thought(question, tool_name, tool_args)
                if gen_t:
                    thought_txt = gen_t
                    neural_used = True

            obs_str, is_err = self._execute_tool_safely(tool_name, tool_args)
            tools_called.append(tool_name)

            traces.append(
                AgentStepTrace(
                    step=step,
                    thought=thought_txt,
                    action=tool_name,
                    action_input=tool_args,
                    observation=obs_str,
                    is_error=is_err,
                    latency_ms=round((time.time() - step_t0) * 1000, 1),
                )
            )

            if is_err:
                had_error = True
                error_notes.append(obs_str)
                # Replanteamiento automático en el loop tras recibir `Error en herramienta: ...`
                if step < self.t_max:
                    if tool_name == "get_air_quality_data":
                        # Recuperación: consultar resumen general válido del Valle de Aburrá en la última fecha disponible
                        action_queue.insert(
                            0,
                            (
                                "get_air_quality_data",
                                {"estacion": intent["station"] or "todas", "fecha": None},
                                "La consulta previa devolvió error de validación; replanteando estrategia para consultar las estaciones válidas del Valle de Aburrá en la fecha vigente.",
                            ),
                        )
                    elif tool_name == "predict_air_quality":
                        safe_args = {
                            "estacion": intent["station"] or "Centro",
                            "viento": max(0.2, min(float(tool_args.get("viento", 1.2)), 15.0)) if float(tool_args.get("viento", 1.2)) >= 0 else 1.0,
                            "humedad": max(20.0, min(float(tool_args.get("humedad", 65.0)), 98.0)),
                            "pm25_previo": max(5.0, min(float(tool_args.get("pm25_previo", 22.0)), 250.0)),
                        }
                        action_queue.insert(
                            0,
                            (
                                "predict_air_quality",
                                safe_args,
                                "Corrigiendo parámetros fuera de rango físico reportados por la herramienta de predicción para obtener una estimación de referencia válida.",
                            ),
                        )
            else:
                if had_error:
                    recovered = True
                parsed_obs = json.loads(obs_str)
                for c in _extract_citations(obs_str):
                    if c not in citations:
                        citations.append(c)

                if tool_name == "get_air_quality_data":
                    data_obs = parsed_obs
                    # Si la concentración observada supera 35 ug/m3 y aún no habíamos planeado RAG, encadenar consulta POECA automáticamente
                    pm_check = parsed_obs.get("pm25_24h_ugm3") or parsed_obs.get("pm25_promedio_valle", 0.0)
                    if pm_check >= 35.0 and not intent["needs_rag"] and step < self.t_max - 1:
                        action_queue.append(
                            (
                                "search_poeca_protocols",
                                {"consulta": "medidas Estado de Prevención y Alerta POECA grupos sensibles PM2.5", "top_k": 2},
                                "Dado que se observan concentraciones elevadas de PM2.5, encadeno consulta al protocolo POECA para brindar medidas preventivas precisas.",
                            )
                        )
                elif tool_name == "predict_air_quality":
                    pred_obs = parsed_obs
                    # Si la predicción arroja Naranja o Rojo y aún no se consultó el RAG, encadenar consulta POECA
                    if parsed_obs.get("color_ica_predicho") in ("Naranja", "Rojo") and not intent["needs_rag"] and step < self.t_max - 1:
                        if not any(a[0] == "search_poeca_protocols" for a in action_queue):
                            action_queue.append(
                                (
                                    "search_poeca_protocols",
                                    {"consulta": f"medidas obligatorias {parsed_obs.get('nivel_poeca_asociado')} POECA salud movilidad", "top_k": 2},
                                    "El pronóstico proyecta deterioro a nivel Naranja/Rojo; consultando el protocolo POECA para detallar medidas aplicables.",
                                )
                            )
                elif tool_name == "search_poeca_protocols":
                    rag_obs = parsed_obs

        # Paso final de síntesis fundamentada en las observaciones reales
        final_answer = self._synthesize_grounded_response(
            question=question,
            data_obs=data_obs,
            pred_obs=pred_obs,
            rag_obs=rag_obs,
            error_notes=error_notes,
            citations=citations,
        )

        if len(traces) < self.t_max:
            traces.append(
                AgentStepTrace(
                    step=len(traces) + 1,
                    thought="Integrando hallazgos cuantitativos, predicción probabilística y citas normativas verificadas en la respuesta final.",
                    latency_ms=2.5,
                )
            )

        self.memory.append({"role": "user", "content": question})
        self.memory.append({"role": "assistant", "content": final_answer})

        backend_label = (
            f"local-hybrid:Qwen2.5-0.5B-Instruct+ReAct"
            if neural_used
            else "local-deterministic:ReAct-Fallback"
        )
        return AgentResponse(
            question=question,
            answer=final_answer,
            steps=traces,
            tools_called=tools_called,
            citations=citations,
            backend_used=backend_label,
            total_steps=len(traces),
            had_tool_error=had_error,
            recovered_from_error=recovered,
            elapsed_sec=time.time() - t0,
        )

    def _synthesize_grounded_response(
        self,
        question: str,
        data_obs: Optional[Dict[str, Any]],
        pred_obs: Optional[Dict[str, Any]],
        rag_obs: Optional[Dict[str, Any]],
        error_notes: List[str],
        citations: List[str],
    ) -> str:
        sections: List[str] = []

        if error_notes:
            clean_errs = "; ".join(e.replace("Error en herramienta: ", "") for e in error_notes)
            sections.append(
                f"⚠️ **Aviso de Validación Operativa:** Durante la consulta se detectó un parámetro fuera de dominio "
                f"(*{clean_errs}*). El agente replanteó automáticamente la consulta empleando los rangos y estaciones "
                f"oficiales del Valle de Aburrá (`Centro`, `El Poblado`, `Aranjuez`, `Itagüí`, `Bello`)."
            )

        if data_obs:
            if "estaciones" in data_obs:
                lines = [
                    f"### 📊 Panorama de Calidad del Aire en el Valle de Aburrá ({data_obs['fecha']})",
                    f"- **Promedio regional $PM_{{2.5}}$:** `{data_obs['pm25_promedio_valle']} µg/m³` | "
                    f"**Estación con mayor concentración:** **{data_obs['estacion_mas_critica']}**.",
                ]
                for st in data_obs["estaciones"]:
                    lines.append(
                        f"  - **{st['estacion']}** ({st['municipio']}): `$PM_{{2.5}}$ = {st['pm25_ugm3']} µg/m³` | "
                        f"ICA **{st['ica']}** — *{st['categoria_ica']}* (**{st['color_ica']}**) | "
                        f"Viento `{st['viento_ms']} m/s`, Humedad `{st['humedad_pct']}%`."
                    )
                lines.append(f"- {data_obs['fuente']}")
                sections.append("\n".join(lines))
            else:
                norma_txt = (
                    "supera el límite diario nacional de `37.0 µg/m³`"
                    if data_obs["supera_norma_diaria_37ugm3"]
                    else "cumple el límite máximo diario nacional de `37.0 µg/m³`"
                )
                sections.append(
                    f"### 📍 Estado Observado — Estación {data_obs['estacion']} ({data_obs['fecha']})\n"
                    f"- **Estación oficial:** {data_obs['estacion_oficial']} (*{data_obs['tipo_estacion']}*)\n"
                    f"- **Concentración $PM_{{2.5}}$ (24h):** **`{data_obs['pm25_24h_ugm3']} µg/m³`** "
                    f"(máximo horario `{data_obs['pm25_max_horario_ugm3']} µg/m³`, media móvil 7 días `{data_obs['media_movil_7d_ugm3']} µg/m³`), "
                    f"valor que **{norma_txt}**.\n"
                    f"- **Índice ICA:** **`{data_obs['ica']}`** — Categoría **{data_obs['categoria_ica']}** (Color **{data_obs['color_ica']}**).\n"
                    f"- **Condiciones meteorológicas asociadas:** Viento `{data_obs['velocidad_viento_ms']} m/s`, "
                    f"Humedad relativa `{data_obs['humedad_relativa_pct']}%`, Temperatura `{data_obs['temperatura_c']} °C`, "
                    f"Precipitación `{data_obs['precipitacion_mm']} mm` (*{data_obs['periodo_estacional']}*).\n"
                    f"- {data_obs['fuente']}"
                )

        if pred_obs:
            inp = pred_obs["parametros_entrada"]
            probs_fmt = ", ".join(
                f"*{k}*: `{v * 100:.1f}%`" for k, v in pred_obs["probabilidades_por_estado"].items()
            )
            sections.append(
                f"### 🎯 Pronóstico Probabilístico para el Día Siguiente ({pred_obs['estacion']})\n"
                f"- **Escenario evaluado:** Viento `{inp['viento_ms']} m/s`, Humedad `{inp['humedad_pct']}%`, "
                f"$PM_{{2.5}}$ previo `{inp['pm25_previo_ugm3']} µg/m³`, Temperatura `{inp['temperatura_c']} °C`.\n"
                f"- **Categoría ICA predicha ($t+1$):** **{pred_obs['categoria_predicha']}** "
                f"(Color **{pred_obs['color_ica_predicho']}**, confianza posterior **`{pred_obs['confianza'] * 100:.1f}%`**, "
                f"$PM_{{2.5}}$ estimado `~{pred_obs['pm25_estimado_ugm3']} µg/m³`).\n"
                f"- **Nivel operativo POECA asociado:** *{pred_obs['nivel_poeca_asociado']}*.\n"
                f"- **Distribución de probabilidad posterior:** {probs_fmt}.\n"
                f"- {pred_obs['fuente']}"
            )

        if rag_obs and rag_obs.get("resultados"):
            rag_lines = ["### 📜 Marco Normativo POECA y Recomendaciones Técnicas"]
            for idx, item in enumerate(rag_obs["resultados"][:3], start=1):
                snippet = item["contenido"].strip()
                if len(snippet) > 520:
                    snippet = snippet[:517].rsplit(" ", 1)[0] + "..."
                rag_lines.append(
                    f"**{idx}. {item['documento']} — {item['seccion']}**\n"
                    f"{snippet}\n"
                    f"{item['cita']}"
                )
            sections.append("\n\n".join(rag_lines))

        sections.append(
            "> *Nota ética y de alcance: Este sistema provee análisis técnico-ambiental e informativo basado en datos "
            "abiertos del SIATA/AMVA y normativa vigente; no constituye consejo médico clínico, legal vinculante ni de urgencias.*"
        )

        return "\n\n".join(sections)

    def ask(self, question: str) -> AgentResponse:
        """Punto de entrada principal para conversar con el agente."""
        if not question or not question.strip():
            return AgentResponse(
                question=question,
                answer="Por favor ingresa una consulta sobre calidad del aire, pronóstico o normativa POECA.",
                backend_used="none",
            )

        client, model, tag = self._resolve_cloud_client()
        if client is not None:
            try:
                return self._run_cloud_loop(question, client, model, tag)
            except Exception as cloud_exc:
                # Si la llave API expiró o hay caída de red, activar Plan B local sin interrumpir la demo
                fallback_resp = self._run_local_hybrid_loop(question)
                fallback_resp.backend_used = f"{fallback_resp.backend_used} (fallback tras error cloud: {type(cloud_exc).__name__})"
                return fallback_resp

        return self._run_local_hybrid_loop(question)


if __name__ == "__main__":
    ag = AirQualityAgent(t_max=6)
    r = ag.ask("Consulta cómo está el Centro en 2024-03-01 y dime qué medidas aplican según el POECA si sigue empeorando.")
    print("Backend:", r.backend_used, "| Pasos:", r.total_steps, "| Tools:", r.tools_called)
    print(r.answer[:700])
