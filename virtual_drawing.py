import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture Video
cap = cv2.VideoCapture(0)

# Wait for the camera to initialize
_, frame = cap.read()
h, w, _ = frame.shape  # Get frame size

# Create a blank canvas
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Color options (BGR format)
colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255)
}
color_keys = list(colors.keys())
selected_color = colors["Blue"]  # Default color
eraser_mode = False  # Eraser OFF initially
thickness = 5  # Default thickness
drawing_enabled = True  # Start with drawing enabled

# Smooth drawing by storing previous points
prev_x, prev_y = None, None

def is_fist(landmarks):
    """ Detect if all fingers (except thumb) are curled into a fist. """
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky fingertips
    finger_joints = [6, 10, 14, 18]  # Lower knuckles of the fingers

    # Check if all fingertips are below their respective knuckles (fingers folded)
    folded_fingers = [landmarks.landmark[finger_tips[i]].y > landmarks.landmark[finger_joints[i]].y for i in range(4)]

    return all(folded_fingers)  # If all fingers are folded, return True (fist detected)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw color palette at the top
    for i, (color_name, bgr) in enumerate(colors.items()):
        x1, x2 = i * 120, (i + 1) * 120
        cv2.rectangle(frame, (x1, 0), (x2, 50), bgr, -1)
        cv2.putText(frame, color_name, (x1 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display Eraser Mode & Drawing Status
    if eraser_mode:
        cv2.putText(frame, "Eraser ON", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if not drawing_enabled:
        cv2.putText(frame, "Drawing Paused (Fist Detected)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Check if user is selecting a color
            if y < 50:  
                selected_index = x // 120  # Determine which color box is clicked
                if selected_index < len(color_keys):
                    selected_color = colors[color_keys[selected_index]]
                    eraser_mode = False  # Turn off eraser if color is selected

            # Check for fist
            drawing_enabled = not is_fist(hand_landmarks)

            # Draw only if previous position is known and drawing is enabled
            if prev_x is not None and prev_y is not None and drawing_enabled:
                color_to_use = (0, 0, 0) if eraser_mode else selected_color  # Black for eraser
                thickness_to_use = 20 if eraser_mode else thickness  # Thicker stroke for erasing

                # Use a smoother drawing effect with anti-aliased lines
                cv2.line(canvas, (prev_x, prev_y), (x, y), color_to_use, thickness_to_use, cv2.LINE_AA)

            prev_x, prev_y = x, y  # Update previous position

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Improve blending of the canvas
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    canvas_bg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    combined = cv2.add(frame_bg, canvas_bg)

    # Show the output
    cv2.imshow("Virtual Drawing", combined)

    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('e'):  # Toggle eraser mode
        eraser_mode = not eraser_mode
    elif key == ord('+'):  # Increase thickness
        thickness = min(thickness + 2, 20)
    elif key == ord('-'):  # Decrease thickness
        thickness = max(thickness - 2, 2)
    elif key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
