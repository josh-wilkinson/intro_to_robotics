import cv2
import numpy as np

def detect_aruco_markers():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Define the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # If markers are detected, draw them
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Print marker information
            print(f"Detected {len(ids)} markers: {ids.flatten()}")
            
            # Draw marker IDs and corners
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                center = corner[0].mean(axis=0).astype(int)
                
                # Draw marker ID
                cv2.putText(frame, f"ID: {marker_id}", 
                           (center[0] - 20, center[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
        
        # Display the resulting frame
        cv2.imshow('ArUco Marker Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            cv2.imwrite('aruco_detection_snapshot.jpg', frame)
            print("Frame saved as 'aruco_detection_snapshot.jpg'")
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("ArUco Marker Detection Script")
    print("Make sure you have a camera connected and OpenCV installed.")
    print("OpenCV version:", cv2.__version__)
    
    try:
        detect_aruco_markers()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure OpenCV is properly installed with ArUco support.")

if __name__ == "__main__":
    main()