import cv2
import mediapipe as mp
import numpy as np
import insightface
import math
import os

# Constants from your notebook
OUTPUT_SIZE = (112, 112)
TARGET_LANDMARKS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 77.7346]
], dtype=np.float32)
EYE_LEFT_INNER = 362
EYE_RIGHT_INNER = 133

class FacePipeline:
    def __init__(self):
        print("Initializing MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            static_image_mode=True  
        )
        
        print("Loading InsightFace ArcFace...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1)
        self.rec_model = self.app.models['recognition']
        print("Models loaded.")

    def process_image(self, image_np):
        """
        Takes a raw numpy image (BGR), performs your alignment logic, 
        and returns the 512-D embedding.
        """
        img_h, img_w = image_np.shape[:2]
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None # No face detected

        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- Your Alignment Logic ---
        left_eye_x = face_landmarks[EYE_LEFT_INNER].x
        right_eye_x = face_landmarks[EYE_RIGHT_INNER].x
        
        # Standardize orientation
        source_indices = [EYE_LEFT_INNER, EYE_RIGHT_INNER, 13] if left_eye_x > right_eye_x else [EYE_RIGHT_INNER, EYE_LEFT_INNER, 13]
        
        source_points = np.array([
            [face_landmarks[i].x * img_w, face_landmarks[i].y * img_h] 
            for i in source_indices
        ], dtype=np.float32)

        transform_matrix = cv2.getAffineTransform(source_points, TARGET_LANDMARKS)
        aligned_face = cv2.warpAffine(
            image_np, transform_matrix, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]), flags=cv2.INTER_LINEAR
        )

        # --- Embedding Generation ---
        aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        aligned_normalized = (aligned_rgb.astype(np.float32) - 127.5) / 128.0
        aligned_transposed = np.transpose(aligned_normalized, (2, 0, 1))
        input_blob = np.expand_dims(aligned_transposed, axis=0)
        
        embedding = self.rec_model.session.run(self.rec_model.output_names, {self.rec_model.input_name: input_blob})[0]
        return embedding.flatten()

    def cosine_similarity(self, emb1, emb2):
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        return dot_product / (norm_emb1 * norm_emb2)