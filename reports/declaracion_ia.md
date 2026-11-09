# Declaración de Uso Ético y Transparente de Inteligencia Artificial Generativa

**Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería**  
**Curso:** SI3003 – Inteligencia Artificial (Periodo 2026-1)  
**Proyecto:** Sistema Inteligente de Vigilancia y Asistencia para la Calidad del Aire en el Valle de Aburrá (SIATA – AMVA)  

---

## 1. Marco del Pacto Pedagógico
En cumplimiento estricto del **Pacto Pedagógico del curso SI3003** y los lineamientos de integridad académica de la Universidad EAFIT, se declara de manera transparente el uso de herramientas de Inteligencia Artificial Generativa tanto en el **proceso de desarrollo de software y documentación** como dentro de la **arquitectura funcional de la Parte 2 del sistema**.

---

## 2. Herramientas de IA Generativa Utilizadas y Propósito

| Categoría de Uso | Herramienta / Modelo | Rol y Propósito Específico en el Proyecto |
| :--- | :--- | :--- |
| **Asistencia en Ingeniería de Software y Pair Programming** | **Google Antigravity (Agente de Codificación asistido por modelos Gemini)** | Estructuración modular del repositorio (`src/`, `notebooks/`, `eval/`, `app.py`), automatización de scripts de descarga desde la API Socrata oficial de `datos.gov.co`, generación de plantillas visuales en Plotly/Matplotlib y revisión de pruebas automatizadas. |
| **Componente Funcional del Sistema (Parte 2 — Loop Agéntico)** | **`Qwen/Qwen2.5-0.5B-Instruct`** (local vía HuggingFace `transformers`) y soporte compatible con OpenAI (**Groq `llama-3.3-70b-versatile` / NVIDIA NIM**) | Razonamiento paso a paso en el loop agéntico (`T_max = 6`), selección de herramientas (`tool calling`) y redacción de respuestas asistidas fundamentadas en las observaciones devueltas por las herramientas. |
| **Representación Semántica para RAG (Parte 2)** | **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** + **`FAISS`** | Generación de vectores densos multilingües de 384 dimensiones para indexación y recuperación semántica de los artículos del Protocolo POECA, Resolución 2254 de 2017 y guías de salud del SIATA. |

---

## 3. Verificación Humana, Comprensión Técnica y Rigor Académico
Se deja constancia expresa de que la asistencia de herramientas generativas **no sustituyó la formulación matemática, validación empírica ni auditoría crítica de los resultados**:

1. **Verificación de Datos Reales Colombianos:** Todos los registros meteorológicos y de $PM_{2.5}$ provienen directamente del portal gubernamental `datos.gov.co` (recurso `g4t8-zkc3` del IDEAM – SISAIRE / AMVA, **775,058 mediciones horarias reales** de 2024–2025). Se auditaron rangos físicos y consistencia con los umbrales legales de la Resolución 2254 de 2017.
2. **Implementación Matemática Propia de la Parte 1:** Los algoritmos `GaussianNaiveBayesScratch` y `GaussianHMMScratch` (estimación de prioris, matrices de transición de Markov $A_{ij}$, algoritmo Forward $\alpha_t$, programación dinámica de Viterbi $\delta_t$ y métricas de evaluación multiclase) están implementados explícitamente con operaciones matriciales en `NumPy`, sin delegar el entrenamiento a librerías de caja negra.
3. **Control de Alucinaciones y Responsabilidad Ética:** El agente está restringido a responder únicamente con base en los resultados de sus tres herramientas verificables, incluyendo citas obligatorias `[Fuente: ...]` y una nota ética explícita que advierte que el sistema no constituye diagnóstico médico ni asesoría jurídica vinculante.

