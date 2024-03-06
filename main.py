import argparse
import cv2
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_filename", help="File name of video to analyze")
    args = parser.parse_args()
    
    logger.info("Loading video file {video_filename}", video_filename = args.video_filename)
    capture = cv2.VideoCapture(args.video_filename)
    if not capture.isOpened():
        logger.error("Could not load file {video_filename}", video_filename=args.video_filename)
        exit(-1)

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

        print(f"Delay {frame_delay}")
        cv2.imshow(f"Frame {frame_idx}", frame)
        cv2.waitKey(frame_delay)

        # if cv2.waitKey(0) & 0xff == ord("q"):
        #     break

    capture.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
