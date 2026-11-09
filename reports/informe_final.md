# Informe Técnico Final: Sistema Inteligente de Vigilancia y Asistencia para la Calidad del Aire en el Valle de Aburrá (SIATA – AMVA)

**Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería**  
**Curso:** SI3003 – Inteligencia Artificial (2026-1) | **Modalidad:** Proyecto Final Integrador  

---

## 1. Definición del Problema y Contexto Local
El Valle de Aburrá (Antioquia, Colombia) es una cuenca interandina estrecha y profunda ($1,300$ a $1,750\text{ m s.n.m.}$ en su base, rodeada por laderas de hasta $2,800\text{ m s.n.m.}$) donde habitan más de cuatro millones de personas en diez municipios. Durante las dos transiciones climáticas anuales de la Zona de Confluencia Intertropical (**febrero–abril** y **octubre–noviembre**), la persistencia de nubes estratiformes bajas inhibe la radiación solar incidente en superficie, impidiendo la ruptura de la inversión térmica matutina y manteniendo la capa límite atmosférica por debajo de las crestas montañosas. Esto atrapa el material particulado fino ($PM_{2.5}$, diámetro aerodinámico $\le 2.5\ \mu m$) emitido por fuentes móviles (~80%) e industriales (~20%).

Aunque el **SIATA** y el **Área Metropolitana del Valle de Aburrá (AMVA)** publican mediciones continuas, persiste una brecha crítica para ciudadanos y tomadores de decisiones: comprender qué implican las concentraciones diarias frente al **Índice de Calidad del Aire (ICA)**, anticipar el estado del día siguiente según el viento y la humedad, y conocer qué medidas obligatorias activa el **Protocolo POECA**. Este proyecto resuelve dicha necesidad integrando de forma no trivial un **modelo probabilístico secuencial implementado desde cero en NumPy (Parte 1)** con un **agente conversacional LLM dotado de RAG normativo y herramientas verificables (Parte 2)** y un **dashboard interactivo**.

---

## 2. Datos Reales de Colombia, Preprocesamiento y Plan B Activo
- **Fuente Oficial Verificada:** Portal Único de Datos Abiertos de Colombia (`datos.gov.co`), Subsistema de Información sobre Calidad del Aire (**SISAIRE – IDEAM**), registros reportados oficialmente por la autoridad ambiental **AMVA / Red SIATA**:
  1. **Recurso `g4t8-zkc3`** (*Calidad del Aire en Colombia*): Se descargaron y auditaron **775,058 mediciones horarias reales** entre el `2024-01-01` y el `2025-12-31` ($PM_{2.5}$, temperatura `TAire2`, humedad relativa `HAire2`, velocidad del viento `VViento` y precipitación `PLiquida`).
  2. **Recurso `kekd-7v7h`** (*Promedio Anual SISAIRE*): Empleado para validación cruzada georreferenciada.
- **Cobertura Geoespacial (5 estaciones en eje Norte → Sur):** **Bello** (*Liceo Fernando Vélez*), **Aranjuez** (*I.E. Ciro Mendía*), **Centro** (*Tráfico Centro – Museo de Antioquia, ZUAP*), **El Poblado** (*I.E. José Félix de Restrepo*) e **Itagüí** (*Casa de Justicia*).
- **Saneamiento y Consolidación Legal (24h):** Conforme a la **Resolución 2254 de 2017 (MinAmbiente)**, se filtraron rangos físicamente válidos y se consolidaron **3,655 registros estación-día** ($731\text{ días} \times 5\text{ estaciones}$), calculando el ICA exacto mediante interpolación lineal por tramos:
  $$I_p = \frac{I_{Hi} - I_{Lo}}{BP_{Hi} - BP_{Lo}} (C_p - BP_{Lo}) + I_{Lo}$$
- **Plan B Reproducible Activo:** El dataset consolidado se almacena localmente en `data/processed/air_quality_aburra.csv` junto con su manifiesto de auditoría (`data/raw/manifest_descarga_sisaire.json`), garantizando operación 100% offline y determinista sin riesgo de caída de APIs externas. El EDA (`notebooks/01_eda_y_datos.ipynb`) evidenció una correlación de persistencia diaria $r = 0.78$, gradiente de acumulación Norte $\rightarrow$ Sur y contraste interanual entre **El Niño 2024** (103 estación-días Naranja y 1 Rojo) y **2025** (11 estación-días Naranja).

---

## 3. Modelo Clásico de la Parte 1: Formulación Propia y Comparación Cuantitativa
En `src/classical_model.py` y `notebooks/02_modelo_clasico.ipynb` se implementaron **íntegramente desde cero en NumPy puro** (sin usar `scikit-learn` ni `hmmlearn`) dos modelos probabilísticos para predecir la categoría ICA del día siguiente $S_{t+1} \in \{\text{Buena}, \text{Aceptable}, \text{Dañina a Grupos Sensibles}, \text{Dañina a la Salud}\}$ a partir de $d = 11$ variables causales ($PM_{2.5}$ actual, rezago $t-1$, media móvil 3d, tendencia, línea base histórica de estación, anomalía local, viento, humedad, temperatura, precipitación e indicador bimodal POECA):

1. **Gaussian Naive Bayes (`GaussianNaiveBayesScratch`):** Estima prioris suavizados $\hat{\pi}_k$ y verosimilitudes gaussianas condicionales $\mathcal{N}(z_{tj} \mid \hat{\mu}_{kj}, \hat{\sigma}^2_{kj} + \epsilon)$ en espacio logarítmico estabilizado con `log-sum-exp` y temperatura de descorrelación $\tau$.
2. **Modelo Oculto de Markov Gaussiano + Filtro Bayesiano Dinámico (`GaussianHMMScratch`):** Estima la distribución inicial $\pi$, la matriz estocástica de transición de primer orden $A_{ij} = P(S_{t+1}=j \mid S_t=i)$, las emisiones gaussianas $B_k(x_t)$, el **Filtrado Recursivo Forward** $\alpha_t(j) \propto B_j(x_t)\sum_i \alpha_{t-1}(i)A_{ij}$, la **Decodificación por Programación Dinámica de Viterbi** y la distribución predictiva a un paso acoplada con calibración log-lineal entrenada por gradiente descendente L2 en NumPy.

### Evaluación con Partición Temporal Estricta (`Train: 2024` [$n=1,830$] $\rightarrow$ `Test Hold-out: 2025` [$n=1,820$])
| Modelo Evaluado (Implementación NumPy) | Accuracy | Macro F1 | Weighted F1 | Brier Score ($\downarrow$) | F1 Sensibles (Naranja) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1: Frecuencia Mayoritaria | 0.7566 | 0.2871 | 0.6518 | 0.3881 | 0.0000 |
| Baseline 2: Persistencia Meteorológica ($\hat{y}_{t+1} = y_t$) | 0.8066 | 0.5263 | 0.8066 | 0.3644 | 0.0909 |
| **Gaussian Naive Bayes (Propio – NumPy)** | 0.7324 | 0.5164 | 0.7501 | 0.3764 | **0.1500** |
| **Gaussian HMM + Filtro Bayesiano (Propio – NumPy)** | **0.8231** | **0.5304** | **0.8169** | **0.2547** | 0.0909 |

**Hallazgos clave:** El modelo propio `GaussianHMMScratch` supera a ambas líneas base en **todas las métricas globales**: mejora en **+6.65 pp** el Accuracy sobre el Baseline Mayoritario y en **+1.65 pp** sobre la exigente Persistencia atmosférica, alcanza el mayor **Macro-F1 (`0.5304`)** y reduce en **30.1%** el error cuadrático probabilístico (**Brier Score `0.2547` vs `0.3644`**). Por su parte, `GaussianNaiveBayesScratch` demuestra mayor sensibilidad pura ante episodios críticos Naranja (`F1 = 0.1500` vs `0.0909`).

---

## 4. Arquitectura del Agente LLM, Motor RAG y Herramientas (Parte 2)
```
Usuario (Notebooks / Dashboard Streamlit)
   │
   ▼
Loop Agéntico ReAct (src/agent_core.py | Cota estricta T_max = 6 | Captura de Excepciones)
   ├──► Tool 1: get_air_quality_data(estacion, fecha)   ──► CSV Consolidado SISAIRE/AMVA (Plan B)
   ├──► Tool 2: predict_air_quality(estacion, meteo...) ──► GaussianHMMScratch (Parte 1 - NumPy)
   └──► Tool 3: search_poeca_protocols(consulta)        ──► FAISS IndexFlatIP + MiniLM-L12 (31 chunks POECA)
```
- **RAG Semántico (`src/rag_engine.py`):** Indexa 4 documentos regulatorios estructurados (`Acuerdo Metropolitano POECA 04/2018`, `Resolución 2254/2017 MinAmbiente`, `Guía de Salud Pública SIATA` y `Dinámica Meteorológica del Valle de Aburrá`) en **31 fragmentos jerárquicos** usando embeddings multilingües `paraphrase-multilingual-MiniLM-L12-v2` + `faiss.IndexFlatIP` y re-ranking híbrido léxico-semántico, retornando citas explícitas `[Fuente: ...]`.
- **Resiliencia y Reproducibilidad LLM:** Toda llamada a herramienta está protegida por intercepción de errores que devuelve `Error en herramienta: <detalle>` como observación al loop para replantear los parámetros automáticamente. El agente soporta endpoints compatibles con OpenAI (`Groq`, `NVIDIA NIM`, `OpenAI`) e incorpora un **motor híbrido local (`Qwen/Qwen2.5-0.5B-Instruct` + planificador ReAct)** que garantiza operación inmediata sin requerir llaves API externas.

---

## 5. Evaluación Experimental del Agente y Limitaciones del Sistema
Se evaluó el agente sobre una suite fija de **15 preguntas** (`eval/test_suite_questions.json` y `notebooks/03_evaluacion_agente.ipynb`) que cubren consultas simples de datos, pronósticos probabilísticos, normativa POECA/salud, encadenamientos compuestos de 2 y 3 herramientas y pruebas de recuperación ante errores inducidos (estación `"Bogotá"` y fecha fuera de rango `"2029-08-15"`):

| Métrica Cuantitativa en Suite Fija ($N = 15$ preguntas) | Resultado Obtenido |
| :--- | :---: |
| **Precisión de Selección e Invocación de Herramientas (Tool-Calling)** | **100.0% (15 / 15)** |
| **Fidelidad de Citación Verificable (`[Fuente: ...]`)** | **100.0% (15 / 15)** |
| **Cumplimiento de Presupuesto `Pasos <= T_max (6)`** | **100.0% (Promedio: 2.87 pasos)** |
| **Recuperación Resiliente ante Excepciones de Herramienta (`Q14`, `Q15`)** | **100.0% (2 / 2)** |
| **Latencia Media por Consulta (Ejecución Local CPU)** | **4.6 – 5.7 s / pregunta** |

### Limitaciones Técnicas, Éticas y Trabajo Futuro
1. **Advertencia Ética y Médica Obligatoria:** El sistema es una herramienta analítica e informativa ambiental; **no reemplaza valoración médica clínica, protocolos de urgencias respiratorias ni los actos administrativos oficiales** publicados por el AMVA en gaceta metropolitana.
2. **No Estacionariedad Climática (ENSO):** El *shift* observado entre El Niño (2024) y La Niña (2025) reduce la frecuencia marginal de alertas Naranja/Roja en años lluviosos; en operación continua se recomienda re-entrenamiento móvil semestral incorporando altura de capa límite de ceilómetros SIATA.
3. **Granularidad Espacial:** Las inferencias representan el entorno de las 5 estaciones monitorizadas y no capturan micro-exposiciones puntuales dentro de túneles viales o recintos intramurales.

