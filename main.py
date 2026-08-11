import json
import pickle

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 모델 파일 불러오기
# =========================================================

MODEL_FILE = "diary_emotion_final.keras"
TOKENIZER_FILE = "tokenizer.pkl"
LABEL_FILE = "emotion_labels.json"
SETTINGS_FILE = "model_settings.json"


print("감정 모델을 불러오는 중입니다...")


model = tf.keras.models.load_model(
    MODEL_FILE
)


with open(
    TOKENIZER_FILE,
    "rb",
) as file:
    tokenizer = pickle.load(file)


with open(
    LABEL_FILE,
    "r",
    encoding="utf-8",
) as file:
    emotion_labels = json.load(file)


with open(
    SETTINGS_FILE,
    "r",
    encoding="utf-8",
) as file:
    settings = json.load(file)


MAX_LENGTH = settings["max_length"]


okt = Okt()


print("모델 로드 완료")
print("감정 목록:", emotion_labels)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Emomemo Emotion API"
)


# Framer에서 요청할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 요청 데이터
# =========================================================

class DiaryRequest(BaseModel):
    text: str


# =========================================================
# 한국어 전처리
# =========================================================

def tokenize_korean(text: str):

    tokens = okt.morphs(
        text,
        norm=True,
        stem=True,
    )

    return " ".join(tokens)


# =========================================================
# 감정 판단
# =========================================================

def predict_emotion(text: str):

    text = text.strip()

    if not text:
        raise ValueError(
            "빈 문장은 분석할 수 없습니다."
        )


    tokenized_text = tokenize_korean(
        text
    )


    sequence = tokenizer.texts_to_sequences(
        [tokenized_text]
    )


    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )


    probabilities = model.predict(
        padded,
        verbose=0,
    )[0]


    emotion_index = int(
        np.argmax(probabilities)
    )


    emotion = emotion_labels[
        emotion_index
    ]


    confidence = float(
        probabilities[
            emotion_index
        ]
    )


    probability_dict = {
        emotion_labels[index]:
        round(
            float(probabilities[index]),
            4,
        )
        for index in range(
            len(emotion_labels)
        )
    }


    return {
        "emotion": emotion,
        "confidence": round(
            confidence,
            4,
        ),
        "probabilities":
            probability_dict,
    }


# =========================================================
# 서버 테스트
# =========================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "message":
            "Emomemo Emotion API is running"
    }


# =========================================================
# Framer가 사용할 주소
# =========================================================

@app.post("/predict")
def predict(
    request: DiaryRequest
):

    try:

        return predict_emotion(
            request.text
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )