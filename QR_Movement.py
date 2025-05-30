import cv2
import time
import numpy as np
from collections import deque


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = cv2.QRCodeDetector()

position_history = deque(maxlen=15)
velocity_history = deque(maxlen=5)
movement = "Initializing..."
last_update_time = time.time()
avg_velocity_x = 0
avg_velocity_y = 0

kalman_gain = 0.2
kalman_x, kalman_y = None, None
process_variance = 1e-5

base_threshold = 5
dynamic_threshold_factor = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    data, bbox, _ = detector.detectAndDecode(blurred)

    if bbox is not None:
        pts = bbox.reshape(-1, 2).astype(int)
        
        for i in range(len(pts)):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,0), 3)
        
        center = np.mean(pts, axis=0)
        center_x, center_y = int(center[0]), int(center[1])
        qr_width = np.max(pts[:,0]) - np.min(pts[:,0])
        
        qr_height = np.max(pts[:,1]) - np.min(pts[:,1])
        
        kalman_x = center_x if kalman_x is None else kalman_x + kalman_gain*(center_x - kalman_x)
        kalman_y = center_y if kalman_y is None else kalman_y + kalman_gain*(center_y - kalman_y)
        
        smoothed_x, smoothed_y = int(kalman_x), int(kalman_y)
        position_history.append((smoothed_x, smoothed_y, current_time))
        
        if len(position_history) >= 2:

            (x1, y1, t1), (x2, y2, t2) = position_history[-2], position_history[-1]
            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                velocity_history.append((vx, vy))
                
                if velocity_history:
                    avg_velocity_x = np.mean([v[0] for v in velocity_history])
                    avg_velocity_y = np.mean([v[1] for v in velocity_history])
                
                x_threshold = base_threshold + (qr_width * dynamic_threshold_factor)
                y_threshold = base_threshold + (qr_height * dynamic_threshold_factor)
                
                x_dir = ""
                y_dir = ""
                
                if avg_velocity_x < -x_threshold:
                    x_dir = "← Left"
                elif avg_velocity_x > x_threshold:
                    x_dir = "→ Right"
                
                if avg_velocity_y < -y_threshold:
                    y_dir = "↑ Up"
                elif avg_velocity_y > y_threshold:
                    y_dir = "↓ Down"
                
                movement = "Stable"
                if x_dir and y_dir:
                    movement = f"{x_dir} + {y_dir}"
                elif x_dir:
                    movement = x_dir
                elif y_dir:
                    movement = y_dir
        
        info_text = f"QR: {data[:10]}..." if data else "QR Detected"
        movement_text = f"Movement: {movement}"
        velocity_text = f"Velocity: X={avg_velocity_x:.1f}, Y={avg_velocity_y:.1f} px/s"
        
        cv2.putText(frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, movement_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(frame, velocity_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
        cv2.circle(frame, (smoothed_x, smoothed_y), 8, (0,0,255), -1)
        
        if "Left" in movement:
            cv2.putText(frame, "←", (smoothed_x-40, smoothed_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        if "Right" in movement:
            cv2.putText(frame, "→", (smoothed_x+40, smoothed_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        if "Up" in movement:
            cv2.putText(frame, "↑", (smoothed_x, smoothed_y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        if "Down" in movement:
            cv2.putText(frame, "↓", (smoothed_x, smoothed_y+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    else:
        movement = "No QR detected"
        position_history.clear()
        velocity_history.clear()
        kalman_x = kalman_y = None
        avg_velocity_x = avg_velocity_y = 0
        cv2.putText(frame, movement, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    fps = 1/(time.time()-current_time) if (time.time()-current_time) > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("QR Movement Tracker (3D)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
