# HELEN

HELEN es un asistente doméstico controlado por gestos compuesto por un backend en Python (Flask + Socket.IO) y una
interfaz web optimizada para ejecutarse en Google Chrome o Chromium. A partir de esta versión se abandona cualquier
flujo de empaquetado en ejecutables: el proyecto se distribuye como código fuente y se ejecuta directamente con Python.

## Documentación principal

- [HELEN – Guía completa para ejecutar en Chrome (Windows)](README-windows-chrome.md)
- [HELEN – Guía completa para ejecutar en Chrome (Linux / Raspberry Pi)](README-linux-rpi-chrome.md)
- [CHANGELOG](CHANGELOG.md)

Cada guía cubre requisitos, instalación, comandos de ejecución, validaciones manuales y solución de problemas específicas
por plataforma. El CHANGELOG detalla cualquier eliminación o movimiento de archivos legacy relacionados con empaquetado o
scripts obsoletos.

## Guía rápida de ejecución (local)

Sigue estos pasos para levantar HELEN desde el código fuente usando **un único entorno virtual**:

1. **Requisitos previos**
   - Python 3.11 con `venv` habilitado.
   - Google Chrome o Chromium (con permisos para acceder a la cámara).
   - Controlador de cámara funcionando (en Linux/Raspberry Pi añade tu usuario al grupo `video`).

2. **Clona el repositorio y entra al proyecto**
   ```bash
   git clone <URL-del-repositorio>
   cd HelenProyecto-main
   ```

3. **Crea y activa el entorno virtual**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

4. **Instala todas las dependencias** (backend, modelo y UI) desde el requirements unificado:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Inicia el backend** (autodetecta cámara si no especificas una):
   ```bash
   python -m backendHelen --host 0.0.0.0 --port 5000
   ```
   Parámetros útiles:
   - `--camera` / `--camera-index`: índice numérico (0,1,2) o ruta `/dev/videoX`; usa `auto` para autodetección.
   - `--camera-backend`: fuerza el backend de captura (p. ej. `dshow`, `v4l2`).
   - `--frame-stride`: procesa un frame de cada _N_ muestras para reducir carga.

6. **Abre la interfaz web** en `http://localhost:5000` desde Chrome/Chromium, concede permiso de cámara y valida que ves el streaming y los controles.

7. **Comprueba el estado** (opcional pero recomendado): visita `http://localhost:5000/health` para revisar cámara, modelo y suscripciones SSE. Revisa la consola donde corre el backend para ver logs o posibles sugerencias de configuración.

8. **Automatiza según tu plataforma**: si prefieres no ejecutar los comandos manualmente, usa los scripts soportados en `scripts/` (`setup-pi.sh`, `run-pi.sh`, `setup-windows.ps1`, `helen-run.ps1`, etc.).

## Gestos soportados por el modelo de video

El backend carga automáticamente el modelo TensorFlow ubicado en
`Hellen_model_RN/video_gesture_model/models/gesture_model_20251106_063546/saved_model.pb`.
Para que las acciones lleguen al frontend:

1. Realiza primero la seña **activar** (se ilumina el anillo y se habilita la ventana de comandos).
2. Dentro de esa ventana, ejecuta una de las siguientes señas:

| Seña (labels.json) | Acción en la UI |
| --- | --- |
| `activar` | Despierta el control gestual (alias de `Start`). |
| `agregar` | Dispara el flujo de “Agregar alarma”. |
| `alarma` | Abre la pestaña de alarmas. |
| `clima` | Lleva a la vista de clima. |
| `configuracion` | Abre el apartado de ajustes/configuración. |
| `dispositivos` | Lleva a la vista de dispositivos IoT. |
| `home` | Regresa a la pantalla principal/inicio. |
| `reloj` | Abre la vista de reloj/temporizador. |
| `tutorial` | Muestra el tutorial interactivo. |

Si no ejecutas una seña válida tras activar, el backend volverá automáticamente al estado inactivo y tendrás que repetir la seña de activación.

## Arquitectura resumida

```
+----------------------+        Eventos / SSE        +-------------------------+
|  Google Chrome /     |  <----------------------->  |  backendHelen.server    |
|  Chromium (Frontend) |                            |  Flask + Socket.IO      |
+----------+-----------+                            +-----------+-------------+
           |  HTTP/WebSocket                                   |
           v                                                   v
   UI, timers, tutoriales                              MediaPipe / OpenCV, cámara
```

- **Frontend**: vive en `helen/` y se sirve directamente desde Flask. Las preferencias de UI (p. ej. color de fondo) se
guardan en `localStorage` para que persistan entre reinicios.
- **Backend**: contenido en `backendHelen/`, expone la API REST, streaming de video y diagnósticos.

## Scripts de apoyo vigentes

Los únicos scripts mantenidos para automatizar la instalación y ejecución son los que residen en `scripts/`:

- `scripts/helen-run.ps1` / `scripts/helen-run.bat`
- `scripts/setup-windows.ps1`
- `scripts/run-windows.ps1`
- `scripts/setup-pi.sh`
- `scripts/run-pi.sh`

El resto de los scripts históricos (`run*.bat`, `run*.sh`) fueron archivados en `legacy/` y no reciben soporte.

## Activos legacy

Todo el material relacionado con empaquetado (PyInstaller, Inno Setup, kioskos heredados, etc.) ahora vive en el
_directorio_ [`legacy/`](legacy/README_legacy.md). Conserva la estructura original únicamente como referencia para equipos
que aún dependan de esos artefactos, pero no forma parte del flujo oficial.

Para cualquier contribución nueva utiliza las guías actualizadas y mantén sincronizados los cambios funcionales entre el
código y la documentación.
