# Reporte Cuantitativo de Evaluación del Agente Ambiental (SIATA / POECA)

- **Total de preguntas evaluadas:** `15`
- **Backend LLM empleado:** `local-hybrid:Qwen2.5-0.5B-Instruct+ReAct`
- **Precisión de Tool-Calling:** **`100.0%`**
- **Fidelidad de Citación de Fuentes (`[Fuente: ...]`):** **`100.0%`**
- **Cumplimiento de Cota `T_max <= 6`:** **`100.0%`** (promedio `2.87` pasos)
- **Resiliencia y Recuperación ante Errores en Tools:** **`100.0%`**
- **Tasa de Éxito End-to-End Global:** **`100.0%`**

## Tabla Detallada por Pregunta de Prueba

| ID | Categoría | Tools Esperadas | Tools Invocadas | Tool OK | Cita OK | Pasos | Resiliencia | Resultado |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Q01** | Consulta de Datos (Tool 1) | `get_air_quality_data` | `get_air_quality_data`, `search_poeca_protocols` | ✅ | ✅ (3) | 3/6 | ✅ | **PASS** |
| **Q02** | Consulta de Datos (Tool 1) | `get_air_quality_data` | `get_air_quality_data` | ✅ | ✅ (1) | 2/6 | ✅ | **PASS** |
| **Q03** | Consulta de Datos Regional (Tool 1) | `get_air_quality_data` | `get_air_quality_data` | ✅ | ✅ (1) | 2/6 | ✅ | **PASS** |
| **Q04** | Predicción Modelo Clásico (Tool 2) | `predict_air_quality` | `get_air_quality_data`, `predict_air_quality`, `search_poeca_protocols` | ✅ | ✅ (4) | 4/6 | ✅ | **PASS** |
| **Q05** | Predicción Modelo Clásico (Tool 2) | `predict_air_quality` | `get_air_quality_data`, `predict_air_quality` | ✅ | ✅ (2) | 3/6 | ✅ | **PASS** |
| **Q06** | Predicción Modelo Clásico (Tool 2) | `predict_air_quality` | `predict_air_quality`, `search_poeca_protocols` | ✅ | ✅ (4) | 3/6 | ✅ | **PASS** |
| **Q07** | Normativa RAG POECA (Tool 3) | `search_poeca_protocols` | `search_poeca_protocols` | ✅ | ✅ (3) | 2/6 | ✅ | **PASS** |
| **Q08** | Normativa RAG Resolución 2254 (Tool 3) | `search_poeca_protocols` | `search_poeca_protocols` | ✅ | ✅ (3) | 2/6 | ✅ | **PASS** |
| **Q09** | Salud Pública RAG (Tool 3) | `search_poeca_protocols` | `search_poeca_protocols` | ✅ | ✅ (3) | 2/6 | ✅ | **PASS** |
| **Q10** | Meteorología RAG SIATA (Tool 3) | `search_poeca_protocols` | `search_poeca_protocols` | ✅ | ✅ (3) | 2/6 | ✅ | **PASS** |
| **Q11** | Compuesta Datos + RAG (Tools 1+3) | `get_air_quality_data`, `search_poeca_protocols` | `get_air_quality_data`, `search_poeca_protocols` | ✅ | ✅ (4) | 3/6 | ✅ | **PASS** |
| **Q12** | Compuesta Predicción + RAG (Tools 2+3) | `predict_air_quality`, `search_poeca_protocols` | `get_air_quality_data`, `predict_air_quality`, `search_poeca_protocols` | ✅ | ✅ (5) | 4/6 | ✅ | **PASS** |
| **Q13** | Compuesta Triple Cadena (Tools 1+2+3) | `get_air_quality_data`, `predict_air_quality`, `search_poeca_protocols` | `get_air_quality_data`, `predict_air_quality`, `search_poeca_protocols` | ✅ | ✅ (5) | 4/6 | ✅ | **PASS** |
| **Q14** | Resiliencia Error Estación Inválida | `get_air_quality_data` | `get_air_quality_data`, `get_air_quality_data` | ✅ | ✅ (1) | 3/6 | ✅ | **PASS** |
| **Q15** | Resiliencia Error Fecha Fuera de Rango + RAG | `get_air_quality_data`, `search_poeca_protocols` | `get_air_quality_data`, `get_air_quality_data`, `search_poeca_protocols` | ✅ | ✅ (4) | 4/6 | ✅ | **PASS** |
