import cv2 as cv
import numpy as np
import onnxruntime as ort


class SFaceORT:
    """
    cv2.FaceRecognizerSF 로직을 onnxruntime-gpu에 최적화하여 구현
    - Align & Crop: cv2.estimateAffinePartial2D, cv.warpAffine 사용
    - Feature Extraction: cv.dnn.blobFromImage 사용 (전처리 통합)
    - Matching: NumPy를 사용한 고속 벡터 연산
    """
    def __init__(self, modelPath, disType=0, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        # ONNX Runtime 세션 초기화
        self._sess = ort.InferenceSession(modelPath, providers=list(providers))
        self._disType = disType
        assert self._disType in [0, 1], "disType must be 0 (Cosine) or 1 (L2-Norm)."

        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128

        # 모델 입/출력 이름 캐싱
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

        # Align & Crop용 표준 랜드마크 좌표
        self._landmark_std = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)

    def _align_crop(self, image, face):
        # 입력 face 배열 확인
        face = np.asarray(face)
        if face.ndim == 1:
            landmarks_data = face[4:14]
        elif face.ndim == 2:
            landmarks_data = face[0, 4:14]
        else:
            raise ValueError(f"Invalid face data shape: {face.shape}")

        landmarks_src = landmarks_data.reshape(5, 2)

        # 입꼬리 순서 재배치: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
        landmarks_reordered = np.array([
            landmarks_src[1],
            landmarks_src[0],
            landmarks_src[2],
            landmarks_src[3],
            landmarks_src[4]
        ], dtype=np.float32)

        # 변환 행렬 계산 및 이미지 warp
        tform = cv.estimateAffinePartial2D(landmarks_reordered, self._landmark_std)[0]
        if tform is None:
            return None
        return cv.warpAffine(image, tform, (112, 112), borderMode=cv.BORDER_CONSTANT)

    def infer(self, image, face):
        """
        주어진 얼굴 영역에서 특징 벡터를 추출
        image: 원본 이미지 (BGR)
        face: YuNet 등에서 검출된 얼굴 정보 (1,15) 또는 (15,)
        """
        # 1. Align & Crop
        aligned_face = self._align_crop(image, face)
        if aligned_face is None:
            return None

        # 2. Feature Extraction
        input_blob = cv.dnn.blobFromImage(
            aligned_face,
            scalefactor=1/255.0,
            mean=(0, 0, 0),
            swapRB=False  # 모델이 BGR 입력을 기대
        )

        feature_raw = self._sess.run([self._output_name], {self._input_name: input_blob})[0]
        feature_norm = feature_raw / np.linalg.norm(feature_raw, axis=1, keepdims=True)
        return feature_norm

    def match(self, image1, face1, image2, face2):
        """
        두 얼굴의 유사도를 측정
        disType=0: 코사인 유사도
        disType=1: L2 거리
        """
        feature1 = self.infer(image1, face1)
        feature2 = self.infer(image2, face2)

        if feature1 is None or feature2 is None:
            return (0.0, 0) if self._disType == 0 else (float('inf'), 0)

        if self._disType == 0:  # 코사인 유사도
            score = np.dot(feature1.flatten(), feature2.flatten())
            is_match = 1 if score >= self._threshold_cosine else 0
        else:  # L2 거리
            score = np.linalg.norm(feature1 - feature2)
            is_match = 1 if score <= self._threshold_norml2 else 0
        return score, is_match


# --- 사용 예 ---
if __name__ == '__main__':
    # YuNet 모델 임포트
    try:
        from yunet_ort_optimized import YuNet
    except ImportError:
        print("오류: 'yunet_ort_optimized.py'를 찾을 수 없습니다.")
        exit()

    # 모델 초기화
    detector = YuNet(modelPath='./data/face_detection_yunet_2023mar.onnx')
    recognizer = SFaceORT(modelPath='./data/face_recognition_sface_2021dec.onnx', disType=0)

    # 이미지 로드
    img1 = cv.imread("face1.jpg")
    img2 = cv.imread("face2.jpg")

    if img1 is None or img2 is None:
        print("오류: 'face1.jpg' 또는 'face2.jpg' 파일을 찾을 수 없습니다.")
        exit()

    # 1. 얼굴 검출
    detector.setInputSize((img1.shape[1], img1.shape[0]))
    faces1 = detector.infer(img1)

    detector.setInputSize((img2.shape[1], img2.shape[0]))
    faces2 = detector.infer(img2)

    if faces1.shape[0] == 0 or faces2.shape[0] == 0:
        print("얼굴을 검출할 수 없습니다.")
    else:
        # 2. 얼굴 매칭 (가장 큰 얼굴끼리 비교)
        face_a = faces1[0:1, :]
        face_b = faces2[0:1, :]

        score, is_match = recognizer.match(img1, face_a, img2, face_b)

        print(f"두 얼굴의 코사인 유사도: {score:.4f}")
        print("결과: 동일 인물입니다." if is_match else "결과: 다른 인물입니다.")
