from dataclasses import dataclass
import joblib
from torchvision.datasets.folder import os
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from cv2.typing import MatLike
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from ultralytics.engine.results import Boxes
import numpy as np
from loguru import logger
from PIL import Image


resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


autoencoder_train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

autoencoder_inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

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
            autoencoder_path: str = "autoencoder.pth",
            clustering_path: str = "kmeans.pkl"
        ) -> None:

        self.detection_model = YOLO("yolov8m.pt")

        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.encoder.eval()
        # self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])

        self.clustering = joblib.load(clustering_path)

        self.accepted_classes = accepted_classes

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

    def _crop_image(self, frame: MatLike, box: Boxes):
        coords = box.xyxy[0]
        x1 = int(coords[0])
        y1 = int(coords[1])
        x2 = int(coords[2])
        y2 = int(coords[3])

        cropped = frame[y1:y2, x1:x2]
        cropped_pil = Image.fromarray(cropped)
        return cropped_pil

    def process_video(self, video_path: str, output_path: str):
        os.makedirs(output_path, exist_ok=True)

        results = self.detection_model.predict(
                video_path, 
                stream=True, 
                classes=self.accepted_classes, 
                device=self.device)

        for result in results:
            boxes = [box for box in result.boxes if box.cls == YOLO_PERSON_CLASS]
            frame = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
            cropped_images = [self._crop_image(frame, box) for box in boxes]
            transformed = [resnet_transform(image) for image in cropped_images]
            transformed = torch.from_numpy(np.array(transformed))
            latent_representations = self.encoder(transformed)
            print(self.clustering.labels_)
            print(latent_representations)
            latent_representations = latent_representations.view(latent_representations.size(0), -1)
            latent_representations = latent_representations.detach().numpy()
            person_classes = self.clustering.predict(latent_representations)
            print(person_classes)

            # frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            # cv2.imshow("", frame)
            # cv2.waitKey(20)

    def extract_person_boxes(self, video_path: str, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)

        results = self.detection_model.predict(
                video_path, 
                stream=True, 
                classes=self.accepted_classes, 
                device=self.device)

        for result in results:
            result.save_crop(output_path, "player")

    def process_frame(self, frame: MatLike) -> MatLike:
        boxes = self._predict_frame(frame)
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
        return frame


