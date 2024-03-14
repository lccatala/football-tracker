import argparse
import time
import cv2
from loguru import logger
from models import Predictor, YOLO_PERSON_CLASS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_filename", help="File name of video to analyze", type=str)
    parser.add_argument("--boxes_output_path", required=False, default="data", help="Directory where to save player cutouts", type=str)
    parser.add_argument("--json_output_path", required=False, default="output.json", help="Json file to store data in", type=str)
    parser.add_argument("--video_output_path", required=False, default="output.mp4", help="Resulting video file", type=str)
    args = parser.parse_args()
    

    predictor = Predictor()
    predictor.process_video(args.video_filename, args.json_output_path, args.video_output_path)


    # frame_rate = capture.get(cv2.CAP_PROP_FPS)
    # print(f"Frame rate {frame_rate}")
    # frame_delay = int(1000 / frame_rate) if frame_rate > 0 else 1
    #
    # frame_idx = 0
    # while True:
    #     frame_idx += 1
    #     ret, frame = capture.read()
    #
    #     if not ret:
    #         logger.info("Finished reading video file")
    #         break
    #
    #     logger.info("Reading frame {frame_idx}", frame_idx=frame_idx)
    #     predictor.process_frame(frame)
    #
    #     cv2.imshow(f"Frame {frame_idx}", frame)
    #     cv2.waitKey(frame_delay)
    #
    #
    # capture.release()
    # cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
