from dataclasses import dataclass
import joblib
from torchvision.datasets.folder import os
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from cv2.typing import MatLike
from cv2 import VideoWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from ultralytics.engine.results import Boxes
import numpy as np
from loguru import logger
from PIL import Image
from sklearn.cluster import KMeans


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


YOLO_PERSON_CLASS = 0
YOLO_BALL_CLASS = 32


@dataclass
class Predictor:
    def __init__(
            self, 
            accepted_classes: list[int] = [YOLO_PERSON_CLASS, YOLO_BALL_CLASS], 
            use_gpu: bool = True,
        ) -> None:

        self.detection_model = YOLO("yolov8m.pt")
        self.clustering = KMeans(n_clusters=2)
        self.accepted_classes = accepted_classes

        self.previous_ball_positions = []

        self.device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"

    def _predict_frame(self, frame: MatLike) -> list:
        results = self.detection_model.predict(
                frame, 
                classes=self.accepted_classes, 
                device=self.device)

        return [
            box
            for result in results 
            for box in result.boxes
        ]

    @dataclass
    class ImageCrop:
        frame: MatLike
        x1: int
        y1: int
        x2: int
        y2: int

        upper_green: np.ndarray
        lower_green: np.ndarray

        def __init__(self, frame: MatLike, positions: tuple[int, int, int, int]) -> None:
            self.frame = frame

            self.x1 = positions[0]
            self.y1 = positions[1]
            self.x2 = positions[2]
            self.y2 = positions[3]

            self.upper_green = np.array([70, 0, 50])
            self.lower_green = np.array([80, 255,120])

        def is_referee(self) -> bool:
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            _, frame_binary = cv2.threshold(frame_gray, 30, 255, cv2.THRESH_BINARY)
            total_pixels = frame_binary.size
            black_pixels = total_pixels - cv2.countNonZero(frame_binary)
            percentage_black = (black_pixels / total_pixels) * 100
            referee = percentage_black > 7.0
            return referee

            
        def __get_max_count_idx(self) -> tuple:
            hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask_lower = cv2.inRange(hsv_frame, np.array([1,1,1]), self.lower_green)
            mask_upper = cv2.inRange(hsv_frame, self.upper_green, np.array([255, 255, 255]))
            mask = cv2.bitwise_or(mask_lower, mask_upper)
            masked_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

            hist = cv2.calcHist([masked_frame], [0, 1, 2], None, [10, 10, 10], [1, 256, 1, 256, 1, 256])
            max_count_idx = np.unravel_index(hist.argmax(), hist.shape)

            return max_count_idx

        def _prevalent_color_hsv(self) -> list[int]:
            max_count_idx = self.__get_max_count_idx()

            b = int(max_count_idx[2] * 32)
            g = int(max_count_idx[1] * 32)
            r = int(max_count_idx[0] * 32)
            bgr = np.uint8([[[b, g, r]]]) #type: ignore
            hsv_color = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0,0].tolist() #type: ignore

            return hsv_color

        def _prevalent_color_bgr(self) -> list[int]:
            max_count_idx = self.__get_max_count_idx()

            b = int(max_count_idx[2] * 32)
            g = int(max_count_idx[1] * 32)
            r = int(max_count_idx[0] * 32)
            bgr_color = [b, g, r]

            return bgr_color

    def _crop_image(self, frame: MatLike, box: Boxes) -> ImageCrop:
        x1, y1, x2, y2 = self._get_box_coords(box)

        cropped = frame[y1:y2, x1:x2]
        image_crop = self.ImageCrop(cropped, (x1, y1, x2, y2))
        return image_crop

    def _get_fifth_video_frames(self, video_path: str) -> tuple[list[np.ndarray], float]:
        capture = cv2.VideoCapture(video_path)
        
        fps = capture.get(cv2.CAP_PROP_FPS)
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = np.zeros((num_frames//5, frame_height, frame_width, 3), dtype=np.uint8)

        for i in range(num_frames):
            ret, frame = capture.read()
            if not ret:
                break
            if i % 5 == 0:
                frames[i//5] = frame
        capture.release()

        # OpenCV's frame count estimation can be wrong, so we trim the unused frames at the end
        last_zeros_position = len(frames)
        for i,f in enumerate(reversed(frames)):
            if np.array_equal(f, np.zeros(f.shape)):
                last_zeros_position = i
            else:
                break
        frames = frames[:len(frames)-last_zeros_position-1]

        return list(frames), fps

    def _count_players(self, frame: MatLike, image_crops: list[ImageCrop]) -> tuple[list, MatLike]:
        REFEREE_IDX = 2
        player_counts = [0, 0, 0] # Team 1, Team 2, Referees
        for image_crop in image_crops:
            color = (0, 255, 0)

            if image_crop.is_referee():
                player_counts[REFEREE_IDX] += 1
                color = (0, 0, 0)
            else:
                color = image_crop._prevalent_color_bgr()
                team_idx = self.clustering.predict([color])[0]
                player_counts[team_idx] += 1

                color = self.clustering.cluster_centers_[team_idx]

            frame = cv2.rectangle(
                        frame,
                        (image_crop.x1, image_crop.y1), 
                        (image_crop.x2, image_crop.y2), 
                        color, 5)
        return player_counts, frame

    def __predict_next_ball_position(self) -> tuple[int, int, int, int]:
        if len(self.previous_ball_positions) == 1:
            return self.previous_ball_positions[0]

        # Last resort, not detecting the ball on the first frame 
        if len(self.previous_ball_positions) == 0:
            return (0,0,0,0)

        curr_position = self.previous_ball_positions[-1]
        prev_position = self.previous_ball_positions[-2]

        distance = (curr_position[0] - prev_position[0], curr_position[1] - prev_position[1])

        next_position = (
            curr_position[0] + distance[0], 
            curr_position[1] + distance[1],
            self.previous_ball_positions[-1][2],
            self.previous_ball_positions[-1][3]
        )
        self.previous_ball_positions.append(next_position)
        return next_position

    def _detect_ball(self, ball_box_candidates: list[Boxes], frame: MatLike) -> tuple[tuple[int, int, int, int], bool]:
        if not ball_box_candidates:
            return self.__predict_next_ball_position(), False

        ball_box = max(ball_box_candidates, key=lambda box: box.conf)#type: ignore
        x1, y1, x2, y2 = self._get_box_coords(ball_box)
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2

        coords = (x, y, 30, 31)
        self.previous_ball_positions.append(coords)

        return coords, True

    def process_video(self, video_path: str, output_path_json: str, output_path_video: str):
        video_frames, fps = self._get_fifth_video_frames(video_path)
        print("Video loaded")

        results = self.detection_model.predict(
                video_frames, 
                stream=True, 
                classes=self.accepted_classes, 
                device=self.device)
        print("Detecting on video frames...")

        output_json_list = []
        output_frame_list = []
        for frame_idx, result in enumerate(results):
            player_boxes = [box for box in result.boxes if box.cls == YOLO_PERSON_CLASS]
            ball_box_candidates = [box for box in result.boxes if box.cls == YOLO_BALL_CLASS]

            frame = result.orig_img
            player_crops = [self._crop_image(frame, box) for box in player_boxes]
            ball_coords, ball_detected = self._detect_ball(ball_box_candidates, frame)
            if ball_detected:
                frame = cv2.circle(frame, (ball_coords[0], ball_coords[1]), 10, (0, 0, 255), thickness=-1)

            # On the first frame, train KMeans to distinguish players by their prevalent color
            if frame_idx == 0:
                colors = [ic._prevalent_color_bgr() for ic in player_crops if not ic.is_referee()]
                self.clustering.fit(colors)

            player_counts, frame = self._count_players(result.orig_img, player_crops)
            cv2.imshow("", frame)
            cv2.waitKey(0)

            output_frame_list.append(frame)
            json_line = f'{{"frame": {frame_idx*5}, "home_team":{player_counts[0]}, "away_team": {player_counts[1]}, "refs": {player_counts[2]}, "ball_loc":{ball_coords}}}'
            output_json_list.append(json_line)

        print("Writing json...")
        output_json = "\n".join(output_json_list)
        with open(output_path_json, "w") as file:
            file.write(output_json)

        print("Saving video...")
        height, width, _ = output_frame_list[0].shape
        writer = VideoWriter(output_path_video, cv2.VideoWriter.fourcc(*"mp4v"), fps//5, (width, height))
        for frame in output_frame_list:
            writer.write(frame)
        writer.release()


    def extract_person_boxes(self, video_path: str, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)

        results = self.detection_model.predict(
                video_path, 
                stream=True, 
                classes=self.accepted_classes, 
                device=self.device)

        for result in results:
            result.save_crop(output_path, "player")

    def _get_box_coords(self, box: Boxes) -> tuple[int, int, int, int]:
        coords = box.xyxy[0]

        x1 = int(coords[0])
        y1 = int(coords[1])
        x2 = int(coords[2])
        y2 = int(coords[3])

        return x1, y1, x2, y2

    def process_frame(self, frame: MatLike) -> MatLike:
        boxes = self._predict_frame(frame)
        ball_coords_2d = (-1, -1)
        for box in boxes:
            x1, y1, x2, y2 = self._get_box_coords(box)

            if box.cls == YOLO_BALL_CLASS:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))

                mean_x = (x1 + x2) // 2
                mean_y = (y1 + y2) // 2
                ball_coords_2d = (mean_x, mean_y)
                logger.debug("Ball position: {ball_coords_2d}", ball_coords_2d=ball_coords_2d)
                cv2.circle(frame, (mean_x, mean_y), 10, (255, 0, 255), thickness=-1)

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
        return frame


