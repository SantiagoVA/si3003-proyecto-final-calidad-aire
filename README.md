# 🌿 Sistema Inteligente de Vigilancia y Asistencia para la Calidad del Aire en el Valle de Aburrá (SIATA – AMVA)

**Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería**  
**Curso:** SI3003 – Inteligencia Artificial (2026-1) | **Proyecto Final Integrador**

Solución integral de Inteligencia Artificial construida con **datos reales verificables de Colombia (`datos.gov.co` — IDEAM SISAIRE / Red SIATA-AMVA, 775,058 registros horarios 2024–2025)** que articula:
1. **Parte 1 (IA Clásica desde cero en NumPy):** Clasificador Bayesiano Ingenuo Gaussiano (`GaussianNaiveBayesScratch`) y Modelo Oculto de Markov Gaussiano con Filtrado Forward, Decodificación de Viterbi y Predictor Bayesiano Dinámico (`GaussianHMMScratch`), validados temporalmente frente a dos líneas base.
2. **Parte 2 (IA Moderna — Agente LLM + RAG + 3 Tools):** Loop agéntico explícito con cota `T_max = 6`, captura resiliente de errores en herramientas, motor RAG vectorial **FAISS** (`paraphrase-multilingual-MiniLM-L12-v2`) sobre el **Protocolo POECA**, **Resolución 2254 de 2017** y guías de salud del **SIATA**, con citas obligatorias `[Fuente: ...]`.
3. **Notebooks Ejecutados de Punta a Punta + Dashboard Interactivo en Streamlit:** Todos los experimentos, derivaciones matemáticas, gráficas y evaluaciones están completamente desarrollados y ejecutados en `notebooks/`, complementados por el dashboard conversacional interactivo en `app.py`.

---

## 📂 Estructura del Repositorio

```text
Entrega final IA/
├── data/
│   ├── raw/                         # Manifiesto y extractos crudos reales descargados de datos.gov.co (SISAIRE-AMVA)
│   ├── processed/                   # Dataset limpio (air_quality_aburra.csv - Plan B activo), modelo clásico e índice FAISS
│   └── docs_rag/                    # Corpus normativo estructurado: Acuerdo POECA, Resolución 2254/2017, Salud y Meteorología
├── src/
│   ├── data_loader.py               # Descarga Socrata datos.gov.co, saneamiento, cálculo ICA y carga local Plan B
│   ├── classical_model.py           # Parte 1: Gaussian Naive Bayes y Gaussian HMM + Viterbi 100% en NumPy puro
│   ├── rag_engine.py                # Parte 2: Chunking semántico jerárquico + FAISS IndexFlatIP + re-ranking híbrido
│   ├── agent_tools.py               # 3 herramientas operativas: datos reales, predicción Parte 1 y consulta RAG POECA
│   └── agent_core.py                # Loop agéntico ReAct con T_max=6, manejo de errores y soporte Cloud / Local LLM
├── notebooks/
│   ├── 01_eda_y_datos.ipynb                         # Fase 1: Auditoría de datos reales, EDA espacial/temporal y corpus RAG
│   ├── 02_modelo_clasico.ipynb                      # Fase 2: Derivación matemática, entrenamiento NumPy, Viterbi y baselines
│   ├── 03_evaluacion_agente.ipynb                   # Fases 3 y 5: RAG FAISS, Tools, Loop T_max=6 y evaluación de 15 preguntas
│   └── 04_dashboard_y_simulacion_interactiva.ipynb  # Fase 4 en Notebook: Mapas, series POECA, simulador y demo conversacional
├── eval/
│   ├── test_suite_questions.json    # Banco fijo de 15 preguntas de prueba con ground truth de herramientas y citas
│   ├── run_evaluation.py            # Script CLI reproducible para evaluación cuantitativa del agente
│   └── evaluation_summary.md        # Tabla Markdown de resultados cuantitativos generada automáticamente
├── reports/
│   ├── informe_final.md             # Informe técnico breve (máx. 4 páginas)
│   └── declaracion_ia.md            # Declaración ética de uso de IA generativa conforme al pacto pedagógico
├── app.py                           # Aplicación principal del Dashboard Interactivo + Chat en Streamlit
├── requirements.txt                 # Dependencias de Python verificadas
└── .env.example                     # Plantilla opcional de variables de entorno (funciona 100% con o sin API keys)
```

---

## 🚀 Instalación y Ejecución Rápida (Reproducible a la Primera)

### 1. Requisitos e Instalación
Se requiere **Python 3.10+**. En una terminal ubicada en la raíz del repositorio:

```bash
pip install -r requirements.txt
```

> **Nota sobre Llaves API y Plan B (Cero Fricción):**  
> El sistema está diseñado para correr **de inmediato sin intervención manual ni llaves API obligatorias**:
> - **Con llave Cloud opcional:** Copie `.env.example` a `.env` y agregue `GROQ_API_KEY`, `NVIDIA_API_KEY` u `OPENAI_API_KEY` (o ingrésela en la barra lateral del Dashboard) para usar endpoints remotos compatibles con OpenAI.
> - **Sin llaves API (Por defecto):** El agente activa automáticamente su motor híbrido local (`Qwen/Qwen2.5-0.5B-Instruct` en PyTorch + planificador ReAct) y carga el dataset histórico consolidado local (`data/processed/air_quality_aburra.csv`).

### 2. Levantar el Dashboard Interactivo en una Sola Línea
```bash
streamlit run app.py
```
Desde la interfaz podrá seleccionar estación (`Centro`, `El Poblado`, `Aranjuez`, `Itagüí`, `Bello`), explorar cualquier fecha entre `2024-01-01` y `2025-12-31`, inspeccionar el mapa interactivo y la serie temporal con umbrales POECA, simular condiciones meteorológicas en vivo y conversar con el agente abriendo el **Visor de Auditoría del Loop Agéntico**.

### 3. Explorar o Re-ejecutar los Notebooks Completos
Todos los cuadernos en `notebooks/` ya incluyen sus celdas, tablas y figuras ejecutadas. Para abrirlos en Jupyter o VS Code, o re-ejecutarlos desde terminal:

```bash
jupyter lab notebooks/
```

### 4. Ejecutar la Suite Cuantitativa de Evaluación del Agente (15 Preguntas)
```bash
python eval/run_evaluation.py
```
Genera el reporte detallado en `eval/evaluation_results.json` y la tabla resumen en `eval/evaluation_summary.md`.

---

## 📊 Resumen de Resultados Principales
- **Parte 1 (`GaussianHMMScratch` en NumPy puro, validación temporal `2024 → 2025`):** **Accuracy = `82.31%`**, **Macro-F1 = `0.5304`**, **Weighted-F1 = `0.8169`** y **Brier Score = `0.2547`**, superando tanto al Baseline Mayoritario (`75.66%` / `0.2871`) como al exigente Baseline de Persistencia Meteorológica (`80.66%` / `0.5263`, con `-30.1%` de error de calibración).
- **Parte 2 (Suite de 15 Preguntas Fijas):** **100% en Tool-Calling**, **100% en Citación Verificable (`[Fuente: ...]`)**, **100% dentro de `T_max <= 6` (`2.87` pasos promedio)** y **100% de recuperación ante errores inducidos en herramientas**.

---

## ⚠️ Limitaciones y Uso Responsable
Esta herramienta tiene fines exclusivamente académicos, analíticos e informativos ambientales. **No reemplaza el diagnóstico ni criterio médico profesional**, ni sustituye los actos administrativos oficiales emitidos por el Área Metropolitana del Valle de Aburrá (AMVA) y las autoridades sanitarias.

