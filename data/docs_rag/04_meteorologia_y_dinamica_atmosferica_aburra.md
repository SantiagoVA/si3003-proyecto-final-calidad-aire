# Dinámica Atmosférica, Meteorología y Red de Monitoreo del Valle de Aburrá
**Fuente Técnica:** Documentación Científica SIATA — Área Metropolitana del Valle de Aburrá (AMVA) y Subsistema SISAIRE (IDEAM)  
**Código Documento:** SIATA-METEO-RED-2025  

---

## Sección 1. Topografía y Fenómeno de Estabilidad Atmosférica en el Valle de Aburrá
El Valle de Aburrá es una cuenca interandina estrecha y profunda ubicada en la Cordillera Central de Colombia ($6^\circ 05'$ a $6^\circ 30'$ N; $75^\circ 19'$ a $75^\circ 40'$ W). El fondo del valle se sitúa entre **$1,300$ y $1,750\text{ m s.n.m.}$**, rodeado por laderas montañosas abruptas que alcanzan entre **$2,500$ y $2,800\text{ m s.n.m.}$** (como el Alto de Santa Elena, Alto de Minas y el Páramo de las Baldías), con anchos transversales de apenas $3$ a $8\text{ km}$ en sectores críticos como Medellín Centro e Itagüí-Sabaneta.

### 1.1. Capa Límite Atmosférica e Inversión Térmica
Durante las madrugadas, el enfriamiento radiativo del suelo genera una **inversión térmica superficial**: el aire cercano a la superficie es más frío y denso que el aire superior, impidiendo los movimientos verticales ascendentes (convección). En días soleados despejados, la radiación solar matutina calienta la superficie entre las 09:00 y 11:00 a. m., rompiendo la inversión térmica y expandiendo la **capa límite convectiva** por encima de las crestas montañosas ($> 1,500\text{ m}$ sobre el fondo del valle), permitiendo que los vientos alisios exporten los contaminantes fuera de la cuenca.  
Por el contrario, durante las transiciones climáticas (**febrero–abril** y **octubre–noviembre**), capas densas de nubes estratiformes bajas bloquean la radiación solar entrante. La capa límite permanece deprimida por debajo de las montañas circundantes, actuando como una **"tapa meteorológica"** que acumula día tras día el $PM_{2.5}$ emitido en el fondo del valle.

## Sección 2. Relaciones Físicas entre Variables Meteorológicas y $PM_{2.5}$
1. **Velocidad del Viento ($VViento$, $m/s$):** Mantiene una **relación inversa** con la concentración de $PM_{2.5}$. Vientos débiles o calmas atmosféricas ($< 1.0\ m/s$) favorecen el estancamiento y acumulación local de aerosoles; vientos sostenidos ($> 2.0\ m/s$, canalizados predominantemente de Norte a Sur a lo largo del eje Barbosa–Bello–Medellín–Itagüí) promueven la ventilación y dispersión horizontal.
2. **Humedad Relativa ($HAire2$, %) y Temperatura ($TAire2$, °C):** Altas humedades relativas ($> 75\%$) combinadas con temperaturas moderadas/bajas y baja radiación solar están fuertemente asociadas a días nublados con inversión térmica persistente y favorecen el **crecimiento higroscópico** y la formación de aerosoles inorgánicos secundarios (nitratos y sulfatos de amonio a partir de precursores gaseosos $NO_x$ y $SO_2$).
3. **Precipitación ($PLiquida$, $mm$):** Su efecto depende de la intensidad y la hora:
   - **Lluvias convectivas intensas nocturnas ($> 10\ mm$):** Producen **lavado atmosférico efectivo (*wet scavenging*)**, arrastrando el material particulado hacia la superficie y reduciendo el $PM_{2.5}$ de la mañana siguiente.
   - **Lloviznas débiles diurnas ($< 3\ mm$) con cielo cubierto:** Pueden empeorar paradójicamente la acumulación al incrementar la humedad relativa y suprimir el calentamiento solar sin lograr barrido húmedo suficiente.
4. **Inercia Temporal ($PM_{2.5, t-1}$):** Debido a la topografía semicerrada, las concentraciones de $PM_{2.5}$ exhiben alta **autocorrelación de primer orden** entre días consecutivos, justificando el modelado probabilístico secuencial (Modelos Ocultos de Markov — HMM y clasificadores bayesianos con rezagos temporales).

## Sección 3. Inventario de Emisiones y Estaciones Representativas del Estudio
De acuerdo con el Inventario de Emisiones Atmosféricas del AMVA (actualización Universidad Pontificia Bolivariana / AMVA):
- **Fuentes Móviles (~79% – 81% del $PM_{2.5}$ primario):** Transporte de carga pesada y volquetas (~42%–50%), motocicletas de 2 y 4 tiempos, buses y automóviles particulares.
- **Fuentes Fijas Industriales (~19% – 21% del $PM_{2.5}$ primario):** Industrias textiles, ladrilleras/cerámicas, alimentos, químicas y metalmecánicas con calderas y hornos.

### Estaciones Clave Analizadas en el Sistema Integrado (SISAIRE / SIATA)
1. **Centro (`TRÁFICO CENTRO` — Medellín, ID SISAIRE 29509, Lat 6.2527, Lon -75.5697):** Estación de **Tráfico Urbano** en el corazón de La Candelaria (cerca del Museo de Antioquia). Registra las concentraciones más elevadas por exposición directa a rutas de buses y flujo vehicular intenso; referencia clave para la ZUAP Centro.
2. **El Poblado (`I.E. JOSÉ FÉLIX` — Medellín Sur, ID SISAIRE 31373, Lat 6.1999, Lon -75.5608):** Estación de **Representatividad Poblacional (Fondo Urbano Residencial)** en la comuna 14 (El Poblado). Presenta niveles moderados modulados por la vegetación y mayor altitud relativa en ladera suroriental.
3. **Aranjuez (`I.E. CIRO MENDIA` — Medellín Nororiente, ID SISAIRE 31364, Lat 6.2901, Lon -75.5558):** Estación **Poblacional Urbana** en el norte de Medellín, influenciada por alta densidad poblacional y corredores hacia Bello/Autopista Norte.
4. **Itagüí (`CASA JUSTICIA` — Sur del Valle de Aburrá, ID SISAIRE 29493, Lat 6.1857, Lon -75.5972):** Estación **Poblacional / Industrial-Urbana** en el corredor sur, receptora de los vientos dominantes Norte-Sur que transportan la pluma urbana desde Bello y Medellín hacia el estrechamiento sur del valle, sumada a actividad industrial y logística local.
5. **Bello (`LICEO FERNANDO VÉLEZ` — Norte del Valle de Aburrá, ID SISAIRE 31362, Lat 6.3374, Lon -75.5677):** Estación **Poblacional Norte**, ubicada aguas arriba en la dirección predominante de los vientos alisios, con mejor ventilación promedio respecto al sur del valle.

