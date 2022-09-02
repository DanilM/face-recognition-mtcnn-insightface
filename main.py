import cv2
from Face_Rec import FaceRecognator

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    # using library:
    # True: InsightFace
    # False: DeepFace
    face_rec = FaceRecognator("E:\Projects\mtcnn+insightface\FacesDB", False)

    while True:
        success, img = cap.read()
        if not success:
            print("failed to grab frame")
            break
        cv2.imshow("test", face_rec.get_new_frame(img))
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            break
