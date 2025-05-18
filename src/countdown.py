import cv2
import time
import numpy as np

def start_countdown(seconds=3):
    # Create a blank white image/frame
    img_height, img_width = 300, 500
    img = 255 * np.ones((img_height, img_width, 3), np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    font_color = (0, 0, 255)  # Red in BGR
    tip_text = "Tip: Sit 40-70cm from the webcam for optimal face tracking."

    for i in range(seconds, 0, -1):
        # Re-create a fresh white image each second
        img = 255 * np.ones((img_height, img_width, 3), np.uint8)

        text = f"Starting in {i}"
        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Calculate centered position
        x = (img_width - text_width) // 2
        y = (img_height + text_height) // 2

        # Draw the centered text
        cv2.putText(img, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Draw tip text below
        cv2.putText(img, tip_text, (20, img_height - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (0, 0, 0), 1, cv2.LINE_AA)

        # Show countdown frame
        cv2.imshow("Countdown", img)
        cv2.waitKey(1000)

    cv2.destroyWindow("Countdown")