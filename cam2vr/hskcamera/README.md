# hskcamera

`hskcamera` is a native Python module for capturing images from a webcam on Windows with low latency.

It is developed in C++ and uses:

- Media Foundation to capture video from the camera.
- TurboJPEG to convert MJPG frames to RGB.
- pybind11 to connect C++ with Python.
- NumPy to provide the frames.

The module keeps only the most recent frame. If Python processes frames more slowly than the camera captures them, older frames are discarded to prevent latency from accumulating.

## Requirements

- Windows 10 or 11.
- 64-bit Python.
- NumPy.
- `hskcamera.pyd`.
- `turbojpeg.dll`.

Install NumPy:

```powershell
py -m pip install numpy
```

The program directory must contain:

```text
Project/
├── hskcamera.pyd
├── turbojpeg.dll
└── main.py
```

## Importing the Module

```python
import hskcamera
```

## Basic Usage

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
        raise RuntimeError("Could not open the camera")

    if not camera.start():
        raise RuntimeError("Could not start capture")

    last_sequence = 0

    for _ in range(20):
        result = camera.wait_for_next_frame(
            last_sequence,
            timeout_ms=1000,
        )

        if result is None:
            print("No new frame was received")
            continue

        frame, sequence, timestamp = result

        last_sequence = sequence

        print(
            "Resolution:", frame.shape,
            "| Type:", frame.dtype,
            "| Sequence:", sequence,
            "| Timestamp:", timestamp,
        )

finally:
    camera.stop()
    camera.close()
```

## Methods

### `open()`

Opens and configures the camera.

```python
camera.open(
    camera_index,
    format_index,
    exposure=-6,
    list_formats=False,
)
```

Parameters:

- `camera_index`: camera index.
- `format_index`: native format index.
- `exposure`: manual exposure value.
- `list_formats`: displays the available formats when set to `True`.

Returns `True` if the camera is opened successfully.

```python
camera.open(
    camera_index=1,
    format_index=312,
    exposure=-6,
)
```

The format index may vary depending on the camera and its driver.

---

### `start()`

Starts the capture thread.

```python
camera.start()
```

Returns `True` if capture starts successfully.

The camera must be open before calling this method.

---

### `stop()`

Stops the capture thread.

```python
camera.stop()
```

---

### `close()`

Closes the camera and releases its resources.

```python
camera.close()
```

---

### `get_latest_frame()`

Immediately returns the latest available frame.

```python
result = camera.get_latest_frame()
```

If no frame is available yet, it returns:

```python
None
```

When a frame is available, it returns:

```python
frame, sequence, timestamp = result
```

`frame` is an RGB NumPy array:

```python
frame.shape
# (height, width, 3)

frame.dtype
# uint8
```

This method may return the same frame multiple times if Python requests frames faster than the camera captures them.

---

### `wait_for_next_frame()`

Waits until a frame newer than the last received frame is available.

```python
result = camera.wait_for_next_frame(
    last_sequence,
    timeout_ms=1000,
)
```

Parameters:

- `last_sequence`: sequence number of the last received frame.
- `timeout_ms`: maximum wait time in milliseconds.

When a new frame arrives:

```python
frame, sequence, timestamp = result
```

If the timeout is reached, it returns:

```python
None
```

## Properties

### `is_open`

Indicates whether the camera is open.

```python
print(camera.is_open)
```

### `is_capturing`

Indicates whether capture is active.

```python
print(camera.is_capturing)
```

### `width`

Width of the active format.

```python
print(camera.width)
```

### `height`

Height of the active format.

```python
print(camera.height)
```

### `captured_frame_count`

Number of frames captured by C++.

```python
print(camera.captured_frame_count)
```

## Frame Format

Frames are provided as NumPy arrays:

```text
Format: RGB
Type: uint8
Dimensions: height × width × 3
```

Example for 1280 × 720:

```python
frame.shape
# (720, 1280, 3)
```

Accessing a pixel:

```python
red = frame[y, x, 0]
green = frame[y, x, 1]
blue = frame[y, x, 2]
```

## Buffer Behavior

The module stores only the most recent frame.

```text
Python receives frame 100
The camera captures 101
The camera captures 102
The camera captures 103
Python requests another frame
Python receives frame 103
```

Intermediate frames are discarded to keep the capture current and prevent accumulated latency.