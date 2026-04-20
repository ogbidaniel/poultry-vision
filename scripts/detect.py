import cv2
import argparse
import supervision as sv
from ultralytics import YOLO

def main(source, model_path, classes):
    # Load the model
    model = YOLO(model_path)
    tracker = sv.ByteTrack()

    # Initialize annotators
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=60)
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Convert source to integer if it's a single digit (for USB cameras)
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Successfully opened video source: {source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pass the classes argument to filter detections at the inference level
        results = model(frame, classes=classes, verbose=False)[0]
        
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Dynamically generate labels using the model's class dictionary
        labels = [
            f"{model.names[class_id]} #{tracker_id}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        # Apply annotations
        annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        cv2.imshow("Poultry Tracking", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live poultry tracking with YOLO and Supervision")
    
    # Define arguments
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Video source: file path, USB cam index (e.g., '0' or '1'), or RTSP stream URL."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/poultry-yolov12n-v1.pt", 
        help="Path to YOLO model."
    )
    parser.add_argument(
        "--classes", 
        type=int, 
        nargs="+", 
        default=None, 
        help="List of class IDs to track (e.g., --classes 0 1 2). Leave empty to track all."
    )
    
    args = parser.parse_args()
    main(args.source, args.model, args.classes)