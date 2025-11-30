import argparse
from ultralytics import YOLO
import cv2
import time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, required=True, help='Path to video file or 0 for webcam')
    p.add_argument('--model', type=str, default='yolov8n.pt')
    p.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    p.add_argument('--save', type=str, default=None, help='Path to save output video')
    p.add_argument('--show', action='store_true', help='Show preview window')
    return p.parse_args()

def main():
    args = parse_args()

    # Load model
    print(f"Loading model: {args.model}...")
    model = YOLO(args.model) 

    # Mở source
    source = args.source
    try:
        source_int = int(source)
        cap = cv2.VideoCapture(source_int)
    except:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Không mở được source:", source)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.save:
        print(f"Recording to: {args.save}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    # --- MAIN LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True, conf=args.conf, classes=[19], verbose=False)

        current_count = 0

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            current_count = len(track_ids)

            for box, track_id in zip(boxes, track_ids):
                if track_id < 0: 
                    continue
                
                x1, y1, x2, y2 = box
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Cow #{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {current_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if args.show:
            cv2.imshow('Cow Counter', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    main()