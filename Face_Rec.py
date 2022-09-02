import cv2
from deepface import DeepFace
from deepface.commons import functions
from insightface.app import FaceAnalysis
from mtcnn import MTCNN
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import normalize


class FaceRecognator:
    def __init__(self, database_path, using_library : bool):
        self.known_face_embenddings = []
        self.known_face_names = []
        self.using_library = using_library
        self.detector = MTCNN()
        if using_library:
            self.init_insight_face_data(database_path)
        else:
            self.init_deep_face_data(database_path)

    def compare(self, embedding1, embedding2, threshold):
        diff = np.subtract(embedding1, embedding2)
        dist = np.sum(np.square(diff), 0)
        if dist < threshold:
            return True
        else:
            return False

    def insight_face_recognition(self, img):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return
        info = self.face_analyser.get(img)
        for face in info:
            for embedding in self.known_face_embenddings:
                if self.compare(face.normed_embedding, embedding, 1):
                    return self.known_face_names[self.known_face_embenddings.index(embedding)]

        return 'Unknown'

    def paint_boxes(self, cords_list, img):
        for face in cords_list:
            top, right, bottom, left = int(face['box'][1] + face['box'][3]), \
                                       int(face['box'][0] + face['box'][2]), \
                                       int(face['box'][1]), int(face['box'][0])
            face_img = img[bottom:top, left:right]
            cv2.rectangle(img, (left, bottom),
                          (right, top),
                          (0, 255, 0), 2)
            if self.using_library:
                name = self.insight_face_recognition(img)
            else:
                name = self.deep_face_recognition(face_img)
            cv2.putText(img, name, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return img

    def get_new_frame(self, img):
        boxes = self.detector.detect_faces(img)
        img = self.paint_boxes(boxes, img)
        return img

    def init_insight_face_data(self, database_path):
        self.face_analyser = FaceAnalysis()
        self.face_analyser.prepare(ctx_id=0)
        for filename in os.listdir(database_path):
            img = cv2.imread(os.path.join(database_path, filename))
            self.known_face_names.append(filename[:-4])
            data = self.face_analyser.get(img)
            if len(data) != 0:
                self.known_face_embenddings.append(data[0].normed_embedding)

    def init_deep_face_data(self, database_path):
        self.models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
        self.backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        for filename in os.listdir(database_path):
                img = Image.open(os.path.join(database_path, filename))
                array = np.asarray(img)
                face = self.detector.detect_faces(array)
                face = face[0]
                top, right, bottom, left = int(face['box'][1] + face['box'][3]), \
                                           int(face['box'][0] + face['box'][2]), \
                                           int(face['box'][1]), int(face['box'][0])
                self.known_face_names.append(filename[:-4])
                embedding = DeepFace.represent(array[bottom:top, left:right], model_name=self.models[5], enforce_detection = False, detector_backend=self.backends[3])
                self.known_face_embenddings.append(embedding/np.linalg.norm(embedding))

    def deep_face_recognition(self, img):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return
        #img = functions.preprocess_face(img, enforce_detection=False)
        embeddings = DeepFace.represent(img_path=img, model_name=self.models[5], detector_backend=self.backends[3],
                                        enforce_detection=False)
        embeddings = embeddings / np.linalg.norm(embeddings)
        for embedding in self.known_face_embenddings:
            if self.compare(embeddings, embedding, 1):
                return self.known_face_names[self.known_face_embenddings.index(embedding)]

        return 'Unknown'