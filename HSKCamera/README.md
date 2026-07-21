# hskcamera

`hskcamera` es un módulo nativo de Python para capturar imágenes de una webcam en Windows con baja latencia.

Está desarrollado en C++ y utiliza:

* Media Foundation para capturar la cámara.
* TurboJPEG para convertir MJPG a RGB.
* pybind11 para conectar C++ con Python.
* NumPy para entregar los frames.

El módulo mantiene únicamente el frame más reciente. Si Python procesa más lentamente que la cámara, los frames antiguos se descartan para evitar acumular retraso.

## Requisitos

* Windows 10 u 11.
* Python de 64 bits.
* NumPy.
* `hskcamera.pyd`.
* `turbojpeg.dll`.

Instalar NumPy:

```powershell
py -m pip install numpy
```

La carpeta del programa debe contener:

```text
Proyecto/
├── hskcamera.pyd
├── turbojpeg.dll
└── main.py
```

## Importar el módulo

```python
import hskcamera
```

## Uso básico

```python
import hskcamera


camera = hskcamera.Camera()

try:
    opened = camera.open(
        camera_index=1,
        format_index=312,
        exposure=-6,
        list_formats=False,
    )

    if not opened:
        raise RuntimeError("No se pudo abrir la cámara")

    if not camera.start():
        raise RuntimeError("No se pudo iniciar la captura")

    last_sequence = 0

    for _ in range(20):
        result = camera.wait_for_next_frame(
            last_sequence,
            timeout_ms=1000,
        )

        if result is None:
            print("No se recibió un frame nuevo")
            continue

        frame, sequence, timestamp = result

        last_sequence = sequence

        print(
            "Resolución:", frame.shape,
            "| Tipo:", frame.dtype,
            "| Secuencia:", sequence,
            "| Timestamp:", timestamp,
        )

finally:
    camera.stop()
    camera.close()
```

## Métodos

### `open()`

Abre y configura la cámara.

```python
camera.open(
    camera_index,
    format_index,
    exposure=-6,
    list_formats=False,
)
```

Parámetros:

* `camera_index`: índice de la cámara.
* `format_index`: índice del formato nativo.
* `exposure`: exposición manual.
* `list_formats`: muestra los formatos disponibles cuando es `True`.

Devuelve `True` si la cámara se abre correctamente.

```python
camera.open(
    camera_index=1,
    format_index=312,
    exposure=-6,
)
```

El índice del formato puede cambiar dependiendo de la cámara y del controlador.

---

### `start()`

Inicia el hilo de captura.

```python
camera.start()
```

Devuelve `True` si la captura comienza correctamente.

La cámara debe estar abierta antes de llamar a este método.

---

### `stop()`

Detiene el hilo de captura.

```python
camera.stop()
```

---

### `close()`

Cierra la cámara y libera sus recursos.

```python
camera.close()
```

---

### `get_latest_frame()`

Devuelve inmediatamente el último frame disponible.

```python
result = camera.get_latest_frame()
```

Si todavía no hay ningún frame, devuelve:

```python
None
```

Cuando existe un frame, devuelve:

```python
frame, sequence, timestamp = result
```

`frame` es un array NumPy RGB:

```python
frame.shape
# (altura, anchura, 3)

frame.dtype
# uint8
```

Este método puede devolver el mismo frame varias veces si Python pregunta más rápido que la cámara.

---

### `wait_for_next_frame()`

Espera hasta que exista un frame posterior al último recibido.

```python
result = camera.wait_for_next_frame(
    last_sequence,
    timeout_ms=1000,
)
```

Parámetros:

* `last_sequence`: secuencia del último frame recibido.
* `timeout_ms`: tiempo máximo de espera en milisegundos.

Cuando llega un frame nuevo:

```python
frame, sequence, timestamp = result
```

Si se supera el tiempo de espera, devuelve:

```python
None
```

## Propiedades

### `is_open`

Indica si la cámara está abierta.

```python
print(camera.is_open)
```

### `is_capturing`

Indica si la captura está activa.

```python
print(camera.is_capturing)
```

### `width`

Anchura del formato activo.

```python
print(camera.width)
```

### `height`

Altura del formato activo.

```python
print(camera.height)
```

### `captured_frame_count`

Cantidad de frames capturados por C++.

```python
print(camera.captured_frame_count)
```

## Formato del frame

Los frames se entregan como arrays NumPy:

```text
Formato: RGB
Tipo: uint8
Dimensiones: altura × anchura × 3
```

Ejemplo para 1280 × 720:

```python
frame.shape
# (720, 1280, 3)
```

Acceso a un píxel:

```python
red = frame[y, x, 0]
green = frame[y, x, 1]
blue = frame[y, x, 2]
```

## Funcionamiento del buffer

El módulo guarda únicamente el frame más reciente.

```text
Python recibe el frame 100
La cámara captura 101
La cámara captura 102
La cámara captura 103
Python solicita otro frame
Python recibe el frame 103
```

Los frames intermedios se descartan para mantener una captura actual y evitar retraso acumulado.
