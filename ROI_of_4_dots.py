import cv2
import numpy as np


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera 1 not available. Switching to Camera 0.")
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def detect_red_dots(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Combine two red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Only significant blobs
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))

    return points, mask

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    if len(pts) != 4:
        return None

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      # Top-left
    ordered[2] = pts[np.argmax(s)]      # Bottom-right
    ordered[1] = pts[np.argmin(diff)]   # Top-right
    ordered[3] = pts[np.argmax(diff)]   # Bottom-left

    return ordered

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    points, red_mask = detect_red_dots(frame)

    # Draw detected red dots
    for (x, y) in points:
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

    # If exactly 4 red dots are detected
    if len(points) == 4:
        corners = order_points(points)
        if corners is not None:
            corners_int = corners.astype(int)
            cv2.polylines(frame, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)

            # Create and apply polygon mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [corners_int], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Extracted ROI", roi)

    # Show result frames
    cv2.imshow("Red Dot Detection", frame)
    cv2.imshow("Red Mask", red_mask)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
