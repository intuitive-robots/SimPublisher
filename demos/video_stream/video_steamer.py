import cv2
from simpub.core.video_streamer import VideoStreamerManager

# using cv2 to capture video stream and display it
def main():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    video_streamer_manager = VideoStreamerManager("192.168.0.117")
    video_streamer = video_streamer_manager.create_streamer("camera_stream", 640, 480)
    video_streamer_1 = video_streamer_manager.create_streamer("camera_stream_1", 640, 480)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture video frame.")
            break
        cv2.imshow('Video Stream', frame)
        video_streamer.update_cv_image(frame)
        video_streamer_1.update_cv_image(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()