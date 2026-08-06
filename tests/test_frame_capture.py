from cam2vr.hskcamera import hskcamera

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

    started = camera.start()

    if not started:
        raise RuntimeError("No se pudo iniciar la captura")

    print("Cámara abierta:", camera.is_open)
    print("Capturando:", camera.is_capturing)
    print("Resolución:", camera.width, "x", camera.height)

    last_sequence = 0

    for index in range(20):
        result = camera.wait_for_next_frame(
            last_sequence=last_sequence,
            timeout_ms=1000,
        )

        if result is None:
            print("Timeout esperando frame")
            continue

        frame, sequence, timestamp = result

        last_sequence = sequence

        print(
            f"Frame {index}:",
            "shape =", frame.shape,
            "dtype =", frame.dtype,
            "sequence =", sequence,
            "timestamp =", timestamp,
        )

finally:
    camera.stop()
    camera.close()