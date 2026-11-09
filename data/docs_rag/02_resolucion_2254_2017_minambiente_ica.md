# Resolución 2254 de 2017 — Norma de Calidad del Aire e Índice ICA en Colombia
**Entidad Emisora:** Ministerio de Ambiente y Desarrollo Sostenible (MADS) — República de Colombia  
**Vigencia:** A partir del 1 de enero de 2018 (con metas intermedias a 2030)  
**Código Documento:** MADS-RES-2254-2017  

---

## Capítulo I. Niveles Máximos Permisibles de Contaminantes Criterio (Artículo 2)
La Resolución 2254 de 2017 adopta la norma nacional de calidad del aire ambiente para proteger la salud humana y el bienestar de la población colombiana. Para el material particulado fino **$PM_{2.5}$** (partículas con diámetro aerodinámico menor o igual a $2.5\ \mu m$), los límites máximos permisibles vigentes son:
- **Tiempo de exposición de 24 horas (promedio diario):** **$37\ \mu g/m^3$** (vigente desde 2018). Nota técnica: El artículo 3 proyecta una meta normativa más estricta de $25\ \mu g/m^3$ en el horizonte de convergencia hacia las Guías de Calidad del Aire de la OMS.
- **Tiempo de exposición anual (promedio aritmético anual):** **$25\ \mu g/m^3$** (con meta objetivo de **$15\ \mu g/m^3$** en el marco del PIGECA y lineamientos nacionales).
- Para material particulado respirable **$PM_{10}$**: máximo permisible diario (24 horas) de **$75\ \mu g/m^3$** y anual de **$50\ \mu g/m^3$**.

## Capítulo IV. Índice de Calidad del Aire — ICA (Artículos 18 a 21)
El Índice de Calidad del Aire (**ICA**) es un indicador adimensional estandarizado entre **0 y 500** que comunica de forma clara el grado de pureza o contaminación atmosférica y sus efectos asociados en la salud humana.

### Artículo 19. Ecuación Oficial de Cálculo del ICA
Para una concentración medida $C_p$ del contaminante $p$ (para $PM_{2.5}$, expresada como promedio de 24 horas redondeado a un decimal en $\mu g/m^3$), el valor del ICA $I_p$ se calcula mediante interpolación lineal por tramos:

$$I_p = \frac{I_{Hi} - I_{Lo}}{BP_{Hi} - BP_{Lo}} \cdot (C_p - BP_{Lo}) + I_{Lo}$$

Donde:
- $I_p$: Índice de Calidad del Aire para el contaminante $p$.
- $C_p$: Concentración truncada o redondeada del contaminante $p$ ($\mu g/m^3$).
- $BP_{Hi}$: Punto de corte de concentración mayor o igual a $C_p$.
- $BP_{Lo}$: Punto de corte de concentración menor o igual a $C_p$.
- $I_{Hi}$: Valor del ICA correspondiente a $BP_{Hi}$.
- $I_{Lo}$: Valor del ICA correspondiente a $BP_{Lo}$.

### Tabla 6 (Artículo 20). Puntos de Corte del ICA para $PM_{2.5}$ (Promedio 24 Horas)
| Categoría ICA | Color Oficial | Rango ICA ($I_{Lo}$ – $I_{Hi}$) | Concentración $PM_{2.5}$ 24h ($BP_{Lo}$ – $BP_{Hi}$, $\mu g/m^3$) | Significado Regulatorio |
| :--- | :--- | :---: | :---: | :--- |
| **Buena** | **Verde** | 0 – 50 | $0.0\text{ – }12.0$ | La contaminación atmosférica supone un riesgo bajo o nulo para la salud. |
| **Aceptable** | **Amarillo** | 51 – 100 | $12.1\text{ – }37.0$ | Cumple el límite diario nacional ($37\ \mu g/m^3$). Posibles síntomas leves solo en individuos extremadamente sensibles. |
| **Dañina a la Salud de Grupos Sensibles** | **Naranja** | 101 – 150 | $37.1\text{ – }55.4$ | Excede la norma diaria nacional. Riesgo incrementado para niños, adultos mayores, gestantes y pacientes cardiorrespiratorios. Umbral de **Prevención POECA**. |
| **Dañina a la Salud** | **Rojo** | 151 – 200 | $55.5\text{ – }150.4$ | Todos los individuos pueden comenzar a experimentar efectos en la salud; los grupos sensibles pueden sufrir efectos graves. Umbral de **Alerta POECA**. |
| **Muy Dañina a la Salud** | **Púrpura** | 201 – 300 | $150.5\text{ – }250.4$ | Estado de alerta sanitaria de emergencia; toda la población tiene alta probabilidad de verse severamente afectada. Umbral de **Emergencia POECA**. |
| **Peligrosa** | **Marrón** | 301 – 500 | $250.5\text{ – }500.4$ | Condición crítica extrema con riesgo agudo severo para toda la población expuesta. |

## Criterios de Validación de Datos en el SISAIRE / IDEAM
De acuerdo con el Protocolo para el Monitoreo y Seguimiento de la Calidad del Aire del MADS e IDEAM (Subsistema de Información sobre Calidad del Aire — **SISAIRE**):
1. Un promedio diario de 24 horas se considera **válido y representativo** cuando cuenta con al menos el **75% de los registros horarios válidos del día** (mínimo **18 horas válidas** de las 24 horas).
2. Las estaciones se clasifican según su entorno en:
   - **Estaciones de Tráfico (ej. Tráfico Centro Medellín):** Miden la exposición directa en cañones urbanos de alto flujo vehicular; reflejan picos agudos pero no se emplean de forma aislada sin las estaciones poblacionales para declarar estados metropolitanos generales, aunque sí rigen medidas focalizadas ZUAP.
   - **Estaciones de Fondo / Representatividad Poblacional (ej. I.E. José Félix en El Poblado, I.E. Ciro Mendía en Aranjuez, Casa de Justicia en Itagüí, Liceo Fernando Vélez en Bello):** Representan la exposición típica promedio de las comunidades residenciales y mixtas del Valle de Aburrá y constituyen la base jurídica para activar los niveles de Prevención, Alerta y Emergencia del POECA.

