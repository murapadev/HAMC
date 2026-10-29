# Auditoría completa — HAMC (Hierarchical Adaptive Model Collapse)

> Fecha: 2026-06-08 · Alcance: seguridad, calidad, rendimiento, correctitud, bugs, tests y arquitectura.
> Ficheros analizados: **23 módulos Python** (todo `hamc/`, `main.py`, `dashboard.py`) + 5 ficheros de tests.
> ~6.500 líneas de código fuente. 68 tests pasando, 0 fallos.

---

## 1. Resumen ejecutivo

HAMC implementa un generador procedural Wave Function Collapse jerárquico (global → intermedio → local)
con backtracking adaptativo, validación de conectividad y un dashboard en vivo por SSE. El núcleo de
generación **funciona** y produce mapas coherentes; los tests pasan.

El problema dominante **no es un bug aislado sino el ratio señal/ruido arquitectónico**: alrededor del
**40 % del código fuente es andamiaje "enterprise" que la tubería real nunca ejecuta**. El pipeline que
de verdad genera mapas (`main.py` / `dashboard.py` → `GlobalGenerator` / `IntermediateGenerator` /
`LocalGenerator` → `Cell` → `shannon_entropy`) **no importa** el contenedor DI, el sistema de plugins,
las factories, las strategies, la `architectural_integration`, el `PatternDetector`, el
`ParallelLocalGenerator` ni las clases `AdaptiveEntropyCalculator` / `MultiScaleEntropyAnalyzer`. Todo
eso solo es alcanzable desde los tests. Esto choca directamente con la convención del proyecto
("Ship early, iterate. Sin over-engineering").

Hallazgos accionables que sí afectan al resultado:

- **CRÍTICO**: el dashboard SSE arranca un hilo de generación **sin límite de concurrencia ni de coste**
  por petición, con `Access-Control-Allow-Origin: *`. Vector de DoS / agotamiento de recursos.
- **ALTO**: `_force_region_placement` queda anulado por una re-inicialización: la lógica de "regenerar
  para garantizar todas las regiones" no surte efecto (mapas globales pueden omitir regiones requeridas).
- **ALTO**: `validate_building` compara `block_type == "comercial"` (español) cuando el resto del código
  usa `"commercial"` (inglés) → la validación de bloques comerciales es código muerto.
- **ALTO**: race conditions reales en `ParallelLocalGenerator` (`PatternDetector` compartido sin lock +
  `np.random` global no thread-safe).
- **MEDIO**: `CompatibilityCache` se instancia en cada generador pero **nunca se usa** en el hot path; los
  grafos de compatibilidad son **asimétricos** y `are_compatible` exige simetría → callejones sin salida
  espurios (p. ej. `oasis`↔`oasis`, `village`↔`residential`).
- **MEDIO**: WFC re-escanea toda la rejilla por iteración (O(N²)) y guarda snapshots completos por cada
  punto de backtrack que nunca se purgan en éxito (O(N²) memoria).

Veredicto: **proyecto sano en su núcleo, sobre-ingenierizado en su periferia.** Prioridad 1 = blindar el
dashboard; prioridad 2 = arreglar los 3 bugs de correctitud; prioridad 3 = podar o cablear el andamiaje.

---

## 2. Issues por severidad

### 🔴 CRÍTICO

#### C-1 · DoS / agotamiento de recursos en el servidor SSE
`dashboard.py:350-389` (`_serve_stream`), `dashboard.py:84-89` (`PARAM_LIMITS`)

Cada `GET /api/stream` lanza inmediatamente un hilo daemon que ejecuta una generación completa
(`_run_generation`), **sin límite de hilos concurrentes ni tope de trabajo total**. Los límites de
parámetros son demasiado generosos:

```python
PARAM_LIMITS = {"width": (1,16), "height": (1,16), "subgrid": (1,8), "local": (2,12)}
```

Con `width=16, height=16, subgrid=8, local=12` la rejilla intermedia es 128×128 = 16.384 bloques, y
cada bloque colapsa hasta 144 celdas → ~2,3 M de celdas locales en una sola petición. Una página web
maliciosa cualquiera (el `Access-Control-Allow-Origin: *` lo permite, ver C-2) puede abrir N `EventSource`
contra `localhost:8765` y saturar CPU/memoria. La cola del broker (`queue.Queue()`, `dashboard.py:43`) es
**ilimitada**: si el cliente se desconecta o va lento, el productor sigue encolando eventos sin bloquear.

**Fix**: (a) semáforo global de generaciones concurrentes (p. ej. máx. 2-4); (b) tope de coste agregado
`final_w * final_h` rechazando peticiones que excedan un presupuesto; (c) `queue.Queue(maxsize=...)` y
detección de desconexión para cancelar el hilo; (d) bajar los límites de `PARAM_LIMITS`.

---

### 🟠 ALTO

#### A-1 · `_force_region_placement` se auto-anula (regiones requeridas pueden faltar)
`global_generator.py:52-58`, `77-110` + `base_generator.py:76-77`

```python
# global_generator.py
if missing_regions:
    self.initialize()
    return self._force_region_placement(missing_regions)   # fuerza celdas...
...
def _force_region_placement(self, missing_regions):
    ...
    self.cells[r][c].possible = {region: 1.0}; self.cells[r][c].collapse()
    return super().collapse()      # <-- base.collapse() vuelve a llamar self.initialize()
```

`BaseGenerator.collapse()` (`base_generator.py:77`) llama `self.initialize()` al entrar, y
`GlobalGenerator.initialize()` **recrea todas las celdas desde cero**, descartando la colocación forzada.
Resultado: la rama de "regenerar para garantizar todas las regiones" no hace nada útil; simplemente lanza
otro WFC normal que puede volver a omitir una región. `validate()` (que exige todas las `RegionConfig.TYPES`)
fallaría, pero `collapse()` ya devolvió `True` antes de validar de nuevo y `main.py` solo registra un
warning y continúa. **Mapas globales pueden quedar sin alguna región requerida de forma silenciosa.**

**Fix**: separar la fase de inicialización de la de colapso (que `collapse()` no re-inicialice si ya hay
celdas), o implementar el forzado sembrando *antes* de un único `_run_collapse()` sin re-`initialize`.

#### A-2 · Validación de bloques comerciales es código muerto (typo es/en)
`validator.py:187`

```python
elif block_type == "comercial":   # español
```

Todo el resto del sistema usa `"commercial"` (inglés): `LocalGenerator.SPECIAL_BLOCK_TYPES`
(`local_generator.py:15`), `TileConfig.BLOCK_TO_TILES` (`tile_config.py:64`), `validate_building` se llama
con `block_type="commercial"`. La rama `"comercial"` nunca se ejecuta → **los bloques comerciales pasan
la validación sin comprobar nada** (no se verifica el mínimo de escaparates). Es un bug de correctitud y
además un fallo silencioso.

**Fix**: `elif block_type == "commercial":`.

#### A-3 · Race conditions en `ParallelLocalGenerator`
`parallel_generator.py:161-187`, `94-95` + `entropy.py:462`

`_apply_pattern_constraints` (ejecutado dentro de los hilos worker) llama
`self.pattern_detector.analyze_grid(...)` y `suggest_value(...)` **sin tomar `self.lock`**, mientras que
la recolección de resultados sí muta `self.pattern_detector` bajo lock (`:94-95`). `PatternDetector`
mantiene estado mutable (`known_patterns`, `value_positions`) → corrupción de datos / lectura inconsistente
entre hilos. Además `weighted_random_choice` usa `np.random.choice` (`entropy.py:462`), que opera sobre el
**RNG global de NumPy, no thread-safe**; varios `LocalGenerator.collapse()` en paralelo pueden corromper
el estado del RNG o producir resultados correlacionados/errores.

**Fix**: dar a cada worker su propia instancia de `PatternDetector` (o proteger todos los accesos con el
lock), y usar `np.random.default_rng()` por hilo en lugar del RNG global.

#### A-4 · `--host 0.0.0.0` + CORS comodín expone la generación a la LAN
`dashboard.py:270, 301, 346, 365` (`Access-Control-Allow-Origin: *`), `dashboard.py:394`

El comentario de `dashboard.py:363-364` afirma "We only serve locally so this is safe", pero `--host`
acepta `0.0.0.0` (`:394`) y el CORS es comodín en todas las respuestas. Combinado con C-1, cualquier host
de la red (o cualquier web vía DNS-rebinding contra `127.0.0.1`) puede disparar generaciones. No hay datos
sensibles, pero sí coste de cómputo.

**Fix**: restringir CORS al origen propio, advertir/avisar al usar `0.0.0.0`, y/o exigir un token de sesión
para `/api/stream`.

---

### 🟡 MEDIO

#### M-1 · `CompatibilityCache` instanciado pero nunca usado
`base_generator.py:21` + `propagate()` de cada generador

`self.cache = CompatibilityCache()` se crea en cada generador, pero ningún `propagate()` lo consulta:
llaman directamente a `TileConfig.are_compatible` / `BlockConfig.are_compatible` / `RegionConfig.are_compatible`.
Toda la maquinaria de TTL/LRU/estadísticas de `compatibility_cache.py` (138 líneas) es **inerte en el hot
path**. O se cablea (las comprobaciones de compatibilidad son el bucle más caliente de la propagación) o
se elimina.

#### M-2 · Grafos de compatibilidad asimétricos con `are_compatible` simétrico
`tile_config.py:256-262`, `block_config.py:74-79`

`are_compatible(a,b)` exige `b ∈ adj[a]  AND  a ∈ adj[b]`. Pero los grafos están escritos a mano y son
asimétricos en varios puntos, generando incompatibilidades espurias que provocan backtracking/fallos:

- `block_config.py`: `"oasis": {"sand","river","scrubland"}` **no se incluye a sí mismo** → dos bloques
  `oasis` adyacentes son "incompatibles" (todos los demás bloques sí se auto-incluyen).
- `"village": {"sand","road"}` es bloque de transición desierto↔ciudad, pero los bloques de ciudad
  (`residential`/`commercial`/`industrial`) **no listan `village`** y `village` no los lista → una celda de
  transición `village` junto a un bloque residencial falla la propagación.
- `tile_config.py`: `"Suelo_madera"` lista `"Pared"`, pero `"Pared"` no lista `"Suelo_madera"` → par
  rechazado pese a la intención.

**Fix**: derivar la simetría automáticamente (construir el grafo simétrico al cargar, o cambiar
`are_compatible` a OR controlado), y añadir un test que verifique simetría del grafo.

#### M-3 · WFC O(N²) en tiempo y memoria
`base_generator.py:124-204` (`_run_collapse`), `:206-255` (`_find_min_entropy_cell`), `:297-314`

- `_find_min_entropy_cell` recorre **toda** la rejilla en cada iteración (O(N) por colapso → O(N²) total).
  Un WFC eficiente mantiene un heap/cola de celdas por entropía.
- `pre_counts` (`:161-166`) vuelve a recorrer toda la rejilla en cada colapso solo para diffear conteos.
- `backtrack_manager.push` recibe un `_snapshot()` (`:297-306`) que es una **copia completa de la rejilla**
  por cada colapso, y los puntos **solo se extraen al haber conflicto**, nunca se purgan en éxito → el heap
  acumula ~N snapshots de tamaño N → **O(N²) memoria**. Para los tamaños del dashboard es tolerable, pero
  `main.py --local` no tiene tope y escala mal.

**Fix**: heap de entropía incremental; snapshots diferenciales o poda del heap; cap explícito de tamaño.

#### M-4 · `_pop_adaptive` y `record_failure`: O(N) sobre el heap
`backtrack_manager.py:115-192`, `:194-221`

`_pop_adaptive` hace `self.heap.remove(point)` (O(N)) + `heapq.heapify` (O(N)); `record_failure` recalcula
tasas de éxito recorriendo el heap y vuelve a `heapify` en **cada** fallo. Combinado con M-3 degrada
notablemente en rejillas grandes o con mucho backtracking.

#### M-5 · Doble (triple) inicialización de generadores
`local_generator.py:24` y `intermediate_generator.py:22` (`self.initialize()` en `__init__`) +
`base_generator.py:77` (`collapse()` vuelve a llamar `initialize()`)

`LocalGenerator`/`IntermediateGenerator` inicializan en el constructor **y** otra vez al colapsar. En local
esto recrea celdas y descarta el colapso estratégico inicial; es desperdicio de trabajo y fuente de
confusión (relacionado con la causa raíz de A-1).

**Fix**: inicializar en un único punto; que `collapse()` no re-inicialice si el estado ya está listo.

#### M-6 · Fallo silencioso en `weighted_random_choice`
`entropy.py:464-466`

```python
except Exception as e:
    # Log the specific error for debugging
    return "ERROR"
```

El comentario promete logging pero **no logea** y `e` queda sin usar. Devuelve el centinela `"ERROR"`, que
`Cell.collapse` propaga como `collapsed_value="ERROR"` (cadena truthy) → `_run_collapse` intenta propagar
un valor inexistente y solo entonces lo detecta como conflicto. Errores reales (NaN, pesos corruptos) se
ocultan.

**Fix**: `self.logger.warning("weighted_random_choice falló: %s", e)` y considerar lanzar excepción en lugar
de un centinela string.

#### M-7 · Estado global mutable compartido (RNG y singletons)
`dashboard.py:116-117`, `di_container.py:338`, `plugin_system.py:447`, `architectural_integration.py:231`

`_run_generation` hace `random.seed()` / `np.random.seed()` **globales**; con `ThreadingHTTPServer`, dos
streams concurrentes interfieren en el estado del RNG (la `seed` de un cliente afecta a otro). Además hay
varios singletons globales (`_container`, `_plugin_manager`, `_hamc_facade`, `_config_instance`) que
dificultan el aislamiento entre tests y la concurrencia.

---

### 🟢 BAJO

- **B-1 · Log injection** (`dashboard.py:276-304`): `msg`/`stack` del cliente se escriben en el log del
  servidor sin sanear saltos de línea → forja de entradas de log. `Content-Length` sin tope (`:280-281`)
  permite reservar memoria arbitraria. Bajo impacto (solo logs locales).
- **B-2 · `_run_collapse` línea 204 `return True` inalcanzable** (dentro de `while True`). Código muerto.
- **B-3 · Recursión en validadores** (`validator.py:205, 225, 252`): `_find_vertical/horizontal_path` y
  `flood_fill` son DFS recursivos; con `main.py --local` grande pueden exceder el límite de recursión.
- **B-4 · `Cell.entropy()` con efecto colateral** (`cell.py:72-77`): muta `entropy_history` en cada
  llamada y `_find_min_entropy_cell` la invoca para cada celda en cada iteración → asignaciones inútiles.
- **B-5 · `BacktrackManager.get_stats`** (`backtrack_manager.py:277`): `sum(..., default=0)` — `sum` no
  acepta `default=`; lanzaría `TypeError` si se invocara con heap no vacío. Función no usada en el pipeline.
- **B-6 · `_find_min_entropy_cell` muta `cell.possible`** mientras "busca" (`base_generator.py:234-237`):
  mezcla responsabilidades (find vs. mutate); difícil de razonar.
- **B-7 · Comentarios "Reduced from 0.2/0.3"** en `validator.py:114,116` referencian valores que ya no
  existen — comment rot.
- **B-8 · Inconsistencia de nombres de tiles es/en** (`Agua`, `Hierba`, `Pared`… en español vs. claves de
  bloque en inglés). Funciona, pero invita a typos como A-2.

---

## 3. Arquitectura

**Diseño de 3 capas WFC (núcleo): sólido.** La jerarquía global→intermedio→local con propagación de
restricciones por capa y costura del tilemap es correcta y legible. El patrón Template Method en
`BaseGenerator.collapse()` (`initialize → _run_collapse → validate`) está bien aplicado y es el corazón
real del sistema.

**Andamiaje (periferia): sobre-ingeniería desconectada.** Confirmado por inspección de imports: la tubería
real (`main.py`, `dashboard.py`) **no usa** ninguno de estos módulos, solo los tests:

| Módulo | Líneas | ¿Usado por el pipeline real? |
|--------|-------:|------------------------------|
| `di_container.py` | 381 | No |
| `plugin_system.py` | 463 | No |
| `strategy_patterns.py` | 386 | No |
| `factory_patterns.py` | 296 | No |
| `architectural_integration.py` | 247 | No |
| `pattern_detector.py` | 228 | Solo desde `parallel_generator` (tampoco usado) |
| `parallel_generator.py` | 212 | No |
| `AdaptiveEntropyCalculator` + `MultiScaleEntropyAnalyzer` (`entropy.py`) | ~360 | No (`Cell` usa el `shannon_entropy` standalone) |

Son **~2.570 líneas (≈40 %)** que añaden superficie de bug, coste de mantenimiento y confusión sin
contribuir al resultado. El `DependencyContainer` además tiene un patrón peligroso: cuando una dependencia
no se resuelve, **inyecta `None` silenciosamente** (`di_container.py:224-235`) en vez de fallar.

**Patrones Factory / Strategy**: la `GeneratorFactory.create` (`factory_patterns.py:124-149`) tiene una
cascada de `try/except TypeError` que llega a fabricar subclases dinámicas y hasta `object.__new__` — es
un *code smell* fuerte (la factory no conoce las firmas reales de sus productos). Las strategies duplican
lógica que ya vive en `BaseGenerator`/`BacktrackManager` sin que nadie las conecte.

**Recomendación arquitectónica**: o se elimina el andamiaje (preferible, alineado con "sin over-engineering"),
o se cablea de verdad detrás de una sola entrada. Mantener código solo-para-tests da una falsa sensación de
cobertura.

---

## 4. Cobertura de tests — huecos

68 tests, todos verdes, pero **sesgados hacia el andamiaje no usado** (DI, factory, plugins, strategies) y
ligeros en el núcleo que sí genera mapas.

No está cubierto y debería estarlo:

1. **`dashboard.py` (0 % funcional)**: ni rutas HTTP, ni `_clip`/`PARAM_LIMITS`, ni `_run_generation`, ni el
   protocolo SSE/patches. Es el binario más expuesto (ver C-1) y no tiene un solo test.
2. **`main.py` / `generate_demo`**: pipeline end-to-end sin test (E2E de las 3 capas + costura del tilemap).
3. **A-1 (`_force_region_placement`)**: ningún test fuerza rejillas pequeñas/sesgadas que disparen la rama
   de regiones faltantes — por eso el bug pasó desapercibido.
4. **A-2 (`validate_building` "commercial")**: no hay test que valide un bloque comercial inválido (un test
   real habría fallado y delatado el typo).
5. **`BacktrackManager`**: `_pop_adaptive`, `record_failure`/`record_success`, límites por nivel,
   `get_problematic_positions`, `get_stats` (que además tiene B-5) — sin cobertura.
6. **Simetría de los grafos de compatibilidad** (M-2): test trivial que recorra `COMPATIBILITY` y verifique
   `b∈adj[a] ⇔ a∈adj[b]`; hoy fallaría y es muy valioso.
7. **Concurrencia de `ParallelLocalGenerator`** (A-3): no hay test que lo ejecute con varios workers reales
   y verifique ausencia de corrupción.
8. **Reproducibilidad por seed**: `--seed` debería garantizar salida determinista; no se testea.
9. **Robustez de entropía**: `weighted_random_choice` con pesos negativos/NaN/sumas ≈0 (camino "ERROR").

---

## 5. Top 10 acciones priorizadas

| # | Prioridad | Acción | Ref. |
|---|-----------|--------|------|
| 1 | 🔴 CRÍTICO | Blindar el SSE: semáforo de concurrencia + presupuesto de coste + `Queue(maxsize)` + bajar `PARAM_LIMITS` | C-1 |
| 2 | 🟠 ALTO | Arreglar `_force_region_placement` (no re-`initialize` en `collapse`); separar init de collapse | A-1, M-5 |
| 3 | 🟠 ALTO | `"comercial"` → `"commercial"` en `validate_building` + test que lo cubra | A-2 |
| 4 | 🟠 ALTO | Eliminar race conditions en `ParallelLocalGenerator` (PatternDetector por-worker, RNG por-hilo) | A-3 |
| 5 | 🟠 ALTO | Restringir CORS al origen propio y avisar al bindear `0.0.0.0` | A-4 |
| 6 | 🟡 MEDIO | Hacer simétricos los grafos de compatibilidad (auto-derivar) + test de simetría | M-2 |
| 7 | 🟡 MEDIO | Decidir el andamiaje: **podar** DI/plugins/strategy/factory/parallel/Adaptive o cablearlo de verdad | §3 |
| 8 | 🟡 MEDIO | Cablear `CompatibilityCache` en `propagate()` o eliminarlo | M-1 |
| 9 | 🟡 MEDIO | Optimizar WFC: heap de entropía + poda de snapshots de backtrack (O(N²)→~O(N log N)) | M-3, M-4 |
| 10 | 🟡 MEDIO | Tests del núcleo: E2E `main.py`, dashboard, reproducibilidad por seed, BacktrackManager | §4 |

---

## 6. Ficheros analizados

**23 módulos fuente** (100 % del código de `hamc/` + `main.py` + `dashboard.py`):

`main.py`, `dashboard.py` · `hamc/config/`: `tile_config.py`, `block_config.py`, `region_config.py`,
`generator_config.py`, `advanced_config.py` · `hamc/core/`: `cell.py`, `entropy.py`, `backtrack_manager.py`,
`compatibility_cache.py`, `validator.py`, `generator_state.py`, `pattern_detector.py`, `strategy_patterns.py`,
`factory_patterns.py`, `di_container.py`, `architectural_integration.py` · `hamc/generators/`:
`base_generator.py`, `global_generator.py`, `intermediate_generator.py`, `local_generator.py`,
`parallel_generator.py` · `hamc/plugins/`: `plugin_system.py` · `hamc/visualization/`: `renderer.py`.

Más los 5 ficheros de tests (`tests/test_*.py`) para el análisis de cobertura.
