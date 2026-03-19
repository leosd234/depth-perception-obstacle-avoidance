import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# ── Configuración ──────────────────────────────────────────────
DEPTH_THRESHOLD_M = 2.0   # distancia máxima en metros
YOLO_MODEL        = "yolov8n.pt"  # modelo ligero, se descarga automático
WINDOW_COLOR      = "RealSense D435i — Color + Detección"
WINDOW_DEPTH      = "RealSense D435i — Profundidad"

def main():
    # Inicializar cámara RealSense
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(config)

    # Alinear profundidad al color
    align = rs.align(rs.stream.color)

    # Cargar modelo YOLO
    model = YOLO(YOLO_MODEL)

    print("Presiona Q para salir.")

    try:
        while True:
            # Capturar frames
            frames       = pipeline.wait_for_frames()
            aligned      = align.process(frames)
            color_frame  = aligned.get_color_frame()
            depth_frame  = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            # Detección YOLO
            results = model(color_img, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id          = int(box.cls[0])
                conf            = float(box.conf[0])
                label           = model.names[cls_id]

                # Distancia al centro del bounding box
                cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
                dist_m   = depth_frame.get_distance(cx, cy)

                # Color según distancia (rojo=cerca, verde=lejos)
                if 0 < dist_m < 1.0:
                    color = (0, 0, 255)   # rojo — peligro
                elif dist_m < DEPTH_THRESHOLD_M:
                    color = (0, 165, 255) # naranja — precaución
                else:
                    color = (0, 255, 0)   # verde — libre

                # Dibujar bounding box y etiqueta
                cv2.rectangle(color_img, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.0%} | {dist_m:.2f}m"
                cv2.putText(color_img, text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Colormap de profundidad para visualización
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.08),
                cv2.COLORMAP_JET
            )

            cv2.imshow(WINDOW_COLOR, color_img)
            cv2.imshow(WINDOW_DEPTH, depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Cámara detenida.")

if __name__ == "__main__":
    main()