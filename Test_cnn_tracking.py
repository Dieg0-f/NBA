import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math
import argparse
from collections import defaultdict
import yolov5



# Configurazione iniziale
class NBACameraCalibration:
    """Classe per la calibrazione della camera e trasformazione delle coordinate"""

    def __init__(self):
        # Dimensioni reali del campo NBA in piedi
        self.court_length = 94.0
        self.court_width = 50.0
        self.homography_matrix = None

    def set_court_reference_points(self, image_points, world_points=None):
        """
        Imposta punti di riferimento per calcolare la matrice di omografia

        Args:
            image_points: Punti nell'immagine (nel sistema di coordinate pixel)
            world_points: Punti nel mondo reale (coordinate del campo in piedi)
                        Se None, usa i quattro angoli del campo
        """
        if world_points is None:
            # Usa gli angoli del campo come riferimento
            world_points = np.array([
                [0, 0],  # Angolo in basso a sinistra
                [self.court_width, 0],  # Angolo in basso a destra
                [self.court_width, self.court_length],  # Angolo in alto a destra
                [0, self.court_length]  # Angolo in alto a sinistra
            ], dtype=np.float32)

        image_points = np.array(image_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(image_points, world_points)

    def pixel_to_court_coords(self, pixel_coords):
        """Converte coordinate pixel in coordinate del campo"""
        if self.homography_matrix is None:
            raise ValueError("Devi prima calibrare la camera con set_court_reference_points")

        # Converti un singolo punto o un array di punti
        if len(pixel_coords.shape) == 1:
            pixel_coords = pixel_coords.reshape(1, 2)

        # Aggiungi una colonna di 1 per la trasformazione omografica
        ones = np.ones((pixel_coords.shape[0], 1))
        pixels_homogeneous = np.hstack((pixel_coords, ones))

        # Applica la trasformazione
        court_coords_homogeneous = np.dot(self.homography_matrix, pixels_homogeneous.T).T

        # Normalizza per ottenere le coordinate reali
        court_coords = court_coords_homogeneous[:, :2] / court_coords_homogeneous[:, 2:]

        return court_coords


class PlayerTracker:
    """Classe per il tracciamento dei giocatori usando YOLO"""

    def __init__(self, confidence=0.3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(device)
        self.model.conf = confidence  # Soglia di confidenza
        self.classes = self.model.names
        self.device = device
        self.tracker = None

        # YOLO è pre-addestrato per rilevare persone (classe 0)
        self.person_class_id = 0
        self.colors = np.random.randint(0, 255, size=(100, 3))  # Colori per i tracker

    def initialize_tracker(self, method="CSRT"):
        """Inizializza il tipo di tracker OpenCV da usare"""

        def get_tracker(tracker_name, fallback=None):
            if hasattr(cv2, tracker_name + "_create"):
                return getattr(cv2, tracker_name + "_create")
            elif fallback and hasattr(cv2, "legacy") and hasattr(cv2.legacy, tracker_name + "_create"):
                return getattr(cv2.legacy, tracker_name + "_create")
            else:
                return None

        trackers = {
            'BOOSTING': get_tracker("TrackerBoosting", "TrackerBoosting"),
            'MIL': get_tracker("TrackerMIL", "TrackerMIL"),
            'KCF': get_tracker("TrackerKCF", "TrackerKCF"),
            'TLD': get_tracker("TrackerTLD", "TrackerTLD"),
            'MEDIANFLOW': get_tracker("TrackerMedianFlow", "TrackerMedianFlow"),
            'CSRT': get_tracker("TrackerCSRT", "TrackerCSRT"),
            'MOSSE': get_tracker("TrackerMOSSE", "TrackerMOSSE")
        }

        self.tracking_method = method
        self.tracker_creator = trackers.get(method.upper())
        if self.tracker_creator is None:
            raise ValueError(f"Metodo di tracking non disponibile: {method}")

    def detect_players(self, frame):
        """Rileva i giocatori in un frame usando YOLO"""
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

        player_bboxes = []
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            # Filtra solo le persone (giocatori)
            if int(class_id) == self.person_class_id and conf > self.model.conf:
                player_bboxes.append((int(x1), int(y1), int(x2), int(y2), conf))

        return player_bboxes

    def get_player_center(self, bbox):
        """Calcola il centro di un bounding box (piedi del giocatore)"""
        x1, y1, x2, y2, _ = bbox
        # Usiamo il punto centrale inferiore come posizione dei piedi
        center_x = (x1 + x2) // 2
        center_y = y2  # Prendiamo il punto più basso del bounding box
        return np.array([center_x, center_y])


class BallTracker:
    """Classe per il tracciamento della palla"""

    def __init__(self):
        # Parametri per il rilevamento della palla (da adattare in base alle caratteristiche del video)
        self.orange_lower = np.array([10, 100, 150])  # HSV range per il colore arancione
        self.orange_upper = np.array([25, 255, 255])
        self.min_ball_radius = 5
        self.max_ball_radius = 30

    def detect_ball(self, frame):
        """Rileva la palla da basket usando il colore e la forma"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_circle = None
        best_radius = 0

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if self.min_ball_radius < radius < self.max_ball_radius:
                area = cv2.contourArea(contour)
                circularity = 4 * np.pi * area / (2 * np.pi * radius) ** 2
                if circularity > 0.7 and radius > best_radius:
                    best_circle = ((int(x), int(y)), int(radius))
                    best_radius = radius

        return best_circle


class ShotAnalyzer:
    """Classe per analizzare i tiri"""

    def __init__(self, calibration):
        self.calibration = calibration
        # Coordinate del canestro (approssimative, da calibrare)
        self.basket_pixel_coords = None  # Da impostare manualmente o rilevare
        self.basket_court_coords = None
        # Soglia di distanza per il rilevamento di un tiro (in piedi)
        self.shot_distance_threshold = 3.0
        # Stato del tiro
        self.shot_in_progress = False
        self.shot_start_frame = 0
        self.shooter_id = None
        self.shot_trajectory = []
        # Archivio dei tiri rilevati
        self.shots = []

    def set_basket_location(self, pixel_coords):
        """Imposta la posizione del canestro in pixel"""
        self.basket_pixel_coords = np.array(pixel_coords)
        if self.calibration.homography_matrix is not None:
            self.basket_court_coords = self.calibration.pixel_to_court_coords(self.basket_pixel_coords)[0]

    def detect_shot_attempt(self, frame_num, ball_location, players_locations):
        """Rileva un tentativo di tiro in base al movimento della palla e alla vicinanza dei giocatori"""
        if ball_location is None:
            return

        ball_pixel_coords = np.array([ball_location[0][0], ball_location[0][1]])

        if self.calibration.homography_matrix is not None:
            ball_court_coords = self.calibration.pixel_to_court_coords(ball_pixel_coords)[0]

            if self.shot_in_progress:
                self.shot_trajectory.append((frame_num, ball_pixel_coords, ball_court_coords))
                if self.basket_court_coords is not None:
                    distance_to_basket = np.linalg.norm(ball_court_coords - self.basket_court_coords)
                    if distance_to_basket < self.shot_distance_threshold:
                        self.finalize_shot(frame_num, True)
                if frame_num - self.shot_start_frame > 30:
                    self.finalize_shot(frame_num, False)
            else:
                for player_id, player_location in players_locations.items():
                    player_pixel_coords = player_location["center"]
                    player_court_coords = self.calibration.pixel_to_court_coords(player_pixel_coords)[0]
                    distance = np.linalg.norm(player_court_coords - ball_court_coords)
                    if distance < 2.0:
                        self.start_shot(frame_num, player_id, ball_pixel_coords, ball_court_coords)
                        break

    def start_shot(self, frame_num, player_id, ball_pixel_coords, ball_court_coords):
        """Inizia a tracciare un potenziale tiro"""
        self.shot_in_progress = True
        self.shot_start_frame = frame_num
        self.shooter_id = player_id
        self.shot_trajectory = [(frame_num, ball_pixel_coords, ball_court_coords)]

    def finalize_shot(self, frame_num, made):
        """Finalizza un tiro registrato e lo salva"""
        shot_info = {
            "start_frame": self.shot_start_frame,
            "end_frame": frame_num,
            "shooter_id": self.shooter_id,
            "trajectory": self.shot_trajectory,
            "made": made
        }
        self.shots.append(shot_info)
        self.shot_in_progress = False
        self.shot_trajectory = []
        self.shooter_id = None

    def get_shots_data(self):
        """Restituisce i dati di tutti i tiri analizzati"""
        return self.shots


class NBAShotAnalysis:
    """Classe principale per l'analisi dei tiri NBA"""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Impossibile aprire il video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Inizializza i componenti
        self.calibration = NBACameraCalibration()
        self.player_tracker = PlayerTracker(confidence=0.4)
        self.player_tracker.initialize_tracker()
        self.ball_tracker = BallTracker()
        self.shot_analyzer = ShotAnalyzer(self.calibration)

        # Per il tracciamento dei giocatori
        self.tracked_players = {}
        self.player_trajectories = defaultdict(list)
        self.next_player_id = 0

        # Per la visualizzazione
        self.show_visualization = False
        self.output_video = None

    def calibrate_camera(self, reference_points=None):
        """Calibra la camera utilizzando punti di riferimento"""
        if reference_points is None:
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Impossibile leggere il frame dal video")
            print(
                "Seleziona 4 punti di riferimento sul campo (angoli del campo in senso orario partendo da in basso a sinistra)")
            reference_points = self._manual_select_points(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.calibration.set_court_reference_points(reference_points)

    def _manual_select_points(self, frame):
        """Permette all'utente di selezionare manualmente i punti di riferimento"""
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Select Reference Points', frame)

        cv2.imshow('Select Reference Points', frame)
        cv2.setMouseCallback('Select Reference Points', mouse_callback)
        while len(points) < 4:
            cv2.waitKey(100)
        cv2.destroyAllWindows()
        return np.array(points)

    def set_basket_location(self, pixel_coords=None):
        """Imposta la posizione del canestro"""
        if pixel_coords is None:
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Impossibile leggere il frame dal video")
            print("Seleziona la posizione del canestro")
            pixel_coords = self._manual_select_points(frame)[0]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.shot_analyzer.set_basket_location(pixel_coords)

    def process_video(self, start_frame=0, end_frame=None, visualize=False, output_path=None):
        """Processa il video ed estrae i dati dei giocatori e dei tiri"""
        self.show_visualization = visualize
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if end_frame is None:
            end_frame = self.total_frames
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        current_frame = start_frame
        pbar = tqdm(total=end_frame - start_frame, desc="Processando video")
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            player_bboxes = self.player_tracker.detect_players(frame)
            self._update_player_tracking(frame, current_frame, player_bboxes)
            ball_location = self.ball_tracker.detect_ball(frame)
            players_locations = {player_id: {"center": self.player_tracker.get_player_center(bbox)}
                                 for player_id, bbox in self.tracked_players.items()}
            self.shot_analyzer.detect_shot_attempt(current_frame, ball_location, players_locations)
            if self.show_visualization or self.output_video is not None:
                visualization_frame = self._visualize_results(frame, ball_location)
                if self.show_visualization:
                    cv2.imshow("NBA Shot Analysis", visualization_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if self.output_video is not None:
                    self.output_video.write(visualization_frame)
            current_frame += 1
            pbar.update(1)
        pbar.close()
        self.cap.release()
        if self.show_visualization:
            cv2.destroyAllWindows()
        if self.output_video is not None:
            self.output_video.release()
        return self._prepare_analysis_results()

    def _update_player_tracking(self, frame, frame_num, player_bboxes):
        """Aggiorna il tracciamento dei giocatori"""
        if not self.tracked_players:
            for bbox in player_bboxes:
                tracker = self.player_tracker.tracker_creator()
                tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                player_id = self.next_player_id
                self.next_player_id += 1
                self.tracked_players[player_id] = bbox
                center = self.player_tracker.get_player_center(bbox)
                if self.calibration.homography_matrix is not None:
                    court_coords = self.calibration.pixel_to_court_coords(center)[0]
                    self.player_trajectories[player_id].append((frame_num, center, court_coords))
        else:
            current_positions = set()
            for player_id, bbox in list(self.tracked_players.items()):
                tracker = self.player_tracker.tracker_creator()
                tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    bbox = (x, y, x + w, y + h, 0.0)
                    self.tracked_players[player_id] = bbox
                    center = self.player_tracker.get_player_center(bbox)
                    current_positions.add((center[0], center[1]))
                    if self.calibration.homography_matrix is not None:
                        court_coords = self.calibration.pixel_to_court_coords(center)[0]
                        self.player_trajectories[player_id].append((frame_num, center, court_coords))
                else:
                    del self.tracked_players[player_id]
            for bbox in player_bboxes:
                center = self.player_tracker.get_player_center(bbox)
                if (center[0], center[1]) not in current_positions:
                    tracker = self.player_tracker.tracker_creator()
                    tracker.init(frame, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    player_id = self.next_player_id
                    self.next_player_id += 1
                    self.tracked_players[player_id] = bbox
                    if self.calibration.homography_matrix is not None:
                        court_coords = self.calibration.pixel_to_court_coords(center)[0]
                        self.player_trajectories[player_id].append((frame_num, center, court_coords))

    def _visualize_results(self, frame, ball_location):
        """Visualizza i risultati del tracciamento su un frame"""
        vis_frame = frame.copy()
        for player_id, bbox in self.tracked_players.items():
            x1, y1, x2, y2, _ = bbox
            center = self.player_tracker.get_player_center(bbox)
            color = tuple(map(int, self.player_tracker.colors[player_id % 100]))
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"P{player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(vis_frame, (center[0], center[1]), 3, color, -1)
        if ball_location is not None:
            center, radius = ball_location
            cv2.circle(vis_frame, center, radius, (0, 165, 255), 2)
            cv2.circle(vis_frame, center, 1, (0, 165, 255), -1)
        if self.shot_analyzer.basket_pixel_coords is not None:
            x, y = self.shot_analyzer.basket_pixel_coords.astype(int)
            cv2.circle(vis_frame, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(vis_frame, "Basket", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if self.shot_analyzer.shot_in_progress:
            shooter_id = self.shot_analyzer.shooter_id
            cv2.putText(vis_frame, f"Shot by P{shooter_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis_frame

    def _prepare_analysis_results(self):
        """Prepara i risultati dell'analisi"""
        players_data = {}
        for player_id, trajectory in self.player_trajectories.items():
            frames, pixel_coords, court_coords = zip(*trajectory)
            players_data[player_id] = {
                "frames": frames,
                "pixel_coords": pixel_coords,
                "court_coords": court_coords
            }
        shots_data = self.shot_analyzer.get_shots_data()
        return {
            "players": players_data,
            "shots": shots_data,
            "total_frames": self.total_frames,
            "fps": self.fps
        }

    def visualize_trajectories(self, court_image_path=None):
        """Visualizza le traiettorie dei giocatori su un'immagine del campo"""
        if court_image_path and os.path.exists(court_image_path):
            court = cv2.imread(court_image_path)
        else:
            court_width_px = 500
            court_height_px = court_width_px * (94.0 / 50.0)
            court = np.ones((int(court_height_px), int(court_width_px), 3), dtype=np.uint8) * 255
            cv2.rectangle(court, (0, 0), (int(court_width_px) - 1, int(court_height_px) - 1), (0, 0, 0), 2)
            cv2.line(court, (0, int(court_height_px / 2)), (int(court_width_px), int(court_height_px / 2)), (0, 0, 0),
                     1)
            cv2.circle(court, (int(court_width_px / 2), int(court_height_px / 2)), int(court_width_px / 10), (0, 0, 0),
                       1)
            basket_area_width = int(court_width_px * (16.0 / 50.0))
            basket_area_height = int(court_height_px * (19.0 / 94.0))
            cv2.rectangle(court,
                          (int(court_width_px / 2) - int(basket_area_width / 2), 0),
                          (int(court_width_px / 2) + int(basket_area_width / 2), basket_area_height),
                          (0, 0, 0), 1)
            cv2.rectangle(court,
                          (int(court_width_px / 2) - int(basket_area_width / 2),
                           int(court_height_px) - basket_area_height),
                          (int(court_width_px / 2) + int(basket_area_width / 2), int(court_height_px)),
                          (0, 0, 0), 1)

        plt.figure(figsize=(12, 20))
        plt.imshow(cv2.cvtColor(court, cv2.COLOR_BGR2RGB))
        scale_x = court.shape[1] / self.calibration.court_width
        scale_y = court.shape[0] / self.calibration.court_length

        # Traiettorie giocatori
        for player_id, data in self.player_trajectories.items():
            _, _, court_coords = zip(*data)
            court_coords = np.array(court_coords)
            img_x = court_coords[:, 0] * scale_x
            img_y = court_coords[:, 1] * scale_y
            plt.plot(img_x, img_y, 'o-', alpha=0.7, linewidth=1, markersize=3, label=f"Player {player_id}")

        # Traiettorie tiri
        shots_data = self.shot_analyzer.get_shots_data()
        for i, shot in enumerate(shots_data):
            trajectory = shot["trajectory"]
            _, _, court_coords = zip(*trajectory)
            court_coords = np.array(court_coords)
            img_x = court_coords[:, 0] * scale_x
            img_y = court_coords[:, 1] * scale_y
            plt.plot(img_x, img_y, 'x--', alpha=0.8, linewidth=1, markersize=4, label=f"Shot {i}")

        plt.legend()
        plt.title("Traiettorie dei giocatori e tiri")
        plt.show()


if __name__ == "__main__":
    video_path = "video/0.mp4"
    output_path = "output/0.mp4"

    analyzer = NBAShotAnalysis(video_path)
    analyzer.calibrate_camera()
    analyzer.set_basket_location()
    results = analyzer.process_video(visualize=True, output_path=output_path)
    analyzer.visualize_trajectories()

