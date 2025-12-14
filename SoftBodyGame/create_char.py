import cv2
import numpy as np
import json

def is_ccw(pts: np.ndarray) -> bool:
    """Return True if points are counter-clockwise"""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += (x1 - x0) * (y1 + y0)
    return s < 0  # negative = CCW

# --- Load image ---
img = cv2.imread("SoftBodyGame/123.png", cv2.IMREAD_UNCHANGED)

# Extract alpha mask or grayscale if no alpha
alpha = img[:, :, 3] if img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = (alpha > 20).astype(np.uint8) * 255

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)

# Simplify polygon
epsilon = 0.5
simplified = cv2.approxPolyDP(cnt, epsilon, True)
points = simplified[:, 0, :].astype(float)

# Center points
points -= points.mean(axis=0)

# Flip Y-axis for world coordinates
points[:, 1] *= -1

# Normalize scale (optional)
points /= np.max(np.linalg.norm(points, axis=1))

# Ensure CCW winding in world coordinates
if not is_ccw(points):
    points = points[::-1]

# --- Visualization for debugging ---
vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Draw original contour in blue
cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 1)

# Draw simplified polygon in red
cv2.polylines(vis, [simplified], isClosed=True, color=(0, 0, 255), thickness=2)

# Draw vertices in green (centered for display)
for p in points:
    disp_p = p.copy()
    # Undo world flip and scale for visualization
    disp_p *= np.max(mask.shape)  # scale up for image
    disp_p[1] *= -1  # undo Y-flip
    disp_p += np.array([mask.shape[1]//2, mask.shape[0]//2])
    cv2.circle(vis, tuple(disp_p.astype(int)), 3, (0, 255, 0), -1)

cv2.imshow("Outline check", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Save to JSON ---
data = {
    "name": "Seal",
    "points": points.tolist()
}

json_path = "SoftBodyGame/Resources/Seal.json"
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(points)} points to {json_path}")
