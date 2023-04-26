import cv2

# Open the camera (0 is the default camera index, change it if you have multiple cameras)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Camera could not be opened.")
else:
    # Counter for saved images
    img_counter = 0

    while True:
        # Capture a single frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture an image.")
            break

        # Display the live camera feed
        cv2.imshow("Camera Preview", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Save the captured frame to an image file
            img_name = f"captured_image_{img_counter}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Image saved as {img_name}")
            img_counter += 1
        elif key == ord('q'):
            # Exit the loop when 'q' key is pressed
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()