import cv2
from ultralytics import YOLO

def run_tracking(video_source="test_video.mp4"):
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting video stream... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 + ByteTrack Perception", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking(video_source=0)
