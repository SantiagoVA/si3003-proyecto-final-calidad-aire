"""
Script CLI de Evaluación Cuantitativa del Agente Ambiental (SI3003 - Fase 5).

Ejecuta automáticamente el banco de 15 preguntas de prueba (`eval/test_suite_questions.json`),
audita precisión de llamada a herramientas (Tool-Calling Recall & Exactness), inclusión de citas
verificables (`[Fuente: ...]`), respeto estricto de `T_max = 6` y recuperación resiliente ante
errores simulados de dominio, exportando resultados JSON y tabla Markdown reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_core import AirQualityAgent  # noqa: E402

SUITE_PATH = PROJECT_ROOT / "eval" / "test_suite_questions.json"
RESULTS_JSON_PATH = PROJECT_ROOT / "eval" / "evaluation_results.json"
RESULTS_MD_PATH = PROJECT_ROOT / "eval" / "evaluation_summary.md"


def evaluate_agent_suite(t_max: int = 6, verbose: bool = True) -> Dict:
    questions: List[Dict] = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    agent = AirQualityAgent(t_max=t_max)

    rows = []
    for item in questions:
        agent.reset_memory()
        resp = agent.ask(item["pregunta"])

        expected_tools = set(item["herramientas_esperadas"])
        called_tools_set = set(resp.tools_called)
        tool_match = expected_tools.issubset(called_tools_set)

        has_citation_tag = len(resp.citations) > 0 and "[Fuente:" in resp.answer
        expected_sources = item.get("fuentes_esperadas_substr", [])
        # Al menos las fuentes esenciales esperadas aparecen citadas en la respuesta
        if item["id"] == "Q10":
            # Acepta cualquiera de los 2 documentos meteorológicos/POECA que abordan bimodalidad
            source_grounded = any(sub in resp.answer for sub in expected_sources)
        else:
            source_grounded = all(sub in resp.answer for sub in expected_sources)
        citation_ok = bool(has_citation_tag and source_grounded)

        within_tmax = resp.total_steps <= t_max
        if item.get("evalua_recuperacion_error", False):
            error_resilience_ok = bool(resp.had_tool_error and resp.recovered_from_error)
        else:
            error_resilience_ok = not resp.had_tool_error

        overall_pass = bool(tool_match and citation_ok and within_tmax and error_resilience_ok)

        rec = {
            "id": item["id"],
            "categoria": item["categoria"],
            "pregunta": item["pregunta"],
            "herramientas_esperadas": item["herramientas_esperadas"],
            "herramientas_invocadas": resp.tools_called,
            "tool_calling_correcto": tool_match,
            "citacion_verificada": citation_ok,
            "num_citas": len(resp.citations),
            "pasos_ejecutados": resp.total_steps,
            "cumple_tmax": within_tmax,
            "tuvo_error_herramienta": resp.had_tool_error,
            "recuperacion_exitosa": resp.recovered_from_error,
            "resiliencia_ok": error_resilience_ok,
            "exito_global": overall_pass,
            "latencia_seg": round(resp.elapsed_sec, 2),
            "backend": resp.backend_used,
            "respuesta_resumen": resp.answer[:320].replace("\n", " ") + "...",
            "traza_pasos": [s.__dict__ for s in resp.steps],
        }
        rows.append(rec)
        if verbose:
            status_icon = "PASS" if overall_pass else "FAIL"
            print(
                f"[{status_icon}] {rec['id']} ({rec['categoria']}) | "
                f"Tools={rec['herramientas_invocadas']} | Citas={rec['num_citas']} | "
                f"Pasos={rec['pasos_ejecutados']}/{t_max} | t={rec['latencia_seg']}s"
            )

    n = len(rows)
    summary_metrics = {
        "total_preguntas": n,
        "tasa_tool_calling_correcto_pct": round(100.0 * sum(r["tool_calling_correcto"] for r in rows) / n, 2),
        "tasa_citacion_verificada_pct": round(100.0 * sum(r["citacion_verificada"] for r in rows) / n, 2),
        "tasa_cumplimiento_tmax_pct": round(100.0 * sum(r["cumple_tmax"] for r in rows) / n, 2),
        "tasa_resiliencia_errores_pct": round(100.0 * sum(r["resiliencia_ok"] for r in rows) / n, 2),
        "tasa_exito_global_end_to_end_pct": round(100.0 * sum(r["exito_global"] for r in rows) / n, 2),
        "pasos_promedio": round(sum(r["pasos_ejecutados"] for r in rows) / n, 2),
        "latencia_promedio_seg": round(sum(r["latencia_seg"] for r in rows) / n, 2),
        "backend_evaluado": rows[0]["backend"] if rows else "N/A",
    }

    output = {"metricas_globales": summary_metrics, "detalle_preguntas": rows}
    RESULTS_JSON_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Construir reporte Markdown tabulado para el informe final
    md_lines = [
        "# Reporte Cuantitativo de Evaluación del Agente Ambiental (SIATA / POECA)",
        "",
        f"- **Total de preguntas evaluadas:** `{n}`",
        f"- **Backend LLM empleado:** `{summary_metrics['backend_evaluado']}`",
        f"- **Precisión de Tool-Calling:** **`{summary_metrics['tasa_tool_calling_correcto_pct']}%`**",
        f"- **Fidelidad de Citación de Fuentes (`[Fuente: ...]`):** **`{summary_metrics['tasa_citacion_verificada_pct']}%`**",
        f"- **Cumplimiento de Cota `T_max <= 6`:** **`{summary_metrics['tasa_cumplimiento_tmax_pct']}%`** (promedio `{summary_metrics['pasos_promedio']}` pasos)",
        f"- **Resiliencia y Recuperación ante Errores en Tools:** **`{summary_metrics['tasa_resiliencia_errores_pct']}%`**",
        f"- **Tasa de Éxito End-to-End Global:** **`{summary_metrics['tasa_exito_global_end_to_end_pct']}%`**",
        "",
        "## Tabla Detallada por Pregunta de Prueba",
        "",
        "| ID | Categoría | Tools Esperadas | Tools Invocadas | Tool OK | Cita OK | Pasos | Resiliencia | Resultado |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        t_exp = ", ".join(f"`{t}`" for t in r["herramientas_esperadas"])
        t_inv = ", ".join(f"`{t}`" for t in r["herramientas_invocadas"])
        md_lines.append(
            f"| **{r['id']}** | {r['categoria']} | {t_exp} | {t_inv} | "
            f"{'✅' if r['tool_calling_correcto'] else '❌'} | "
            f"{'✅' if r['citacion_verificada'] else '❌'} ({r['num_citas']}) | "
            f"{r['pasos_ejecutados']}/{t_max} | "
            f"{'✅' if r['resiliencia_ok'] else '❌'} | "
            f"**{'PASS' if r['exito_global'] else 'FAIL'}** |"
        )

    RESULTS_MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    res = evaluate_agent_suite(t_max=6, verbose=True)
    print("\nResumen Global:", json.dumps(res["metricas_globales"], indent=2, ensure_ascii=False))

