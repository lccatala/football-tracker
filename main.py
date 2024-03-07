import argparse
from dataclasses import dataclass
import cv2
from cv2.typing import MatLike
from loguru import logger
from ultralytics import YOLO

YOLO_PERSON_CLASS = 0
YOLO_BALL_CLASS = 32

@dataclass
class Predictor:
    def __init__(self, accepted_classes: list[int]) -> None:
        self.model = YOLO("yolov8m.pt")
        self.accepted_classes = accepted_classes

    def _predict(self, frame: MatLike) -> list:
        results = self.model.predict(frame, classes=self.accepted_classes)
        return [
            box
            for result in results 
            for box in result.boxes
        ]

    def process_frame(self, frame: MatLike) -> None:
        boxes = self._predict(frame)
        ball_coords_2d = (-1, -1)
        for box in boxes:
            coords = box.xyxy[0]
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])

            if box.cls == YOLO_BALL_CLASS:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))

                mean_x = (x1 + x2) // 2
                mean_y = (y1 + y2) // 2
                ball_coords_2d = (mean_x, mean_y)
                logger.debug("Ball position: {ball_coords_2d}", ball_coords_2d=ball_coords_2d)
                cv2.circle(frame, (mean_x, mean_y), 10, (255, 0, 255), thickness=-1)

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_filename", help="File name of video to analyze")
    args = parser.parse_args()
    
    logger.info("Loading video file {video_filename}", video_filename = args.video_filename)
    capture = cv2.VideoCapture(args.video_filename)
    if not capture.isOpened():
        logger.error("Could not load file {video_filename}", video_filename=args.video_filename)
        exit(-1)

    predictor = Predictor([YOLO_PERSON_CLASS, YOLO_BALL_CLASS])

    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate {frame_rate}")
    frame_delay = int(1000 / frame_rate) if frame_rate > 0 else 1

    frame_idx = 0
    while True:
        frame_idx += 1
        ret, frame = capture.read()

        if not ret:
            logger.info("Finished reading video file")
            break

        logger.info("Reading frame {frame_idx}", frame_idx=frame_idx)
        predictor.process_frame(frame)

        cv2.imshow(f"Frame {frame_idx}", frame)
        cv2.waitKey(frame_delay)


    capture.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
