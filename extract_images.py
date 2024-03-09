import argparse
import time
import cv2
from loguru import logger
from predictor import Predictor, YOLO_PERSON_CLASS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_filename", help="File name of video to analyze", type=str)
    parser.add_argument("boxes_output_path", help="Directory where to save player cutouts", type=str)
    args = parser.parse_args()
    
    predictor = Predictor(accepted_classes=[YOLO_PERSON_CLASS])

    predictor.extract_person_boxes(args.video_filename, args.boxes_output_path)
        

if __name__ == "__main__":
    main()
