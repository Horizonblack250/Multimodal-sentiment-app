
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO
from deepface import DeepFace
import torch
import tempfile

# Load models
bert_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
bert_model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
yolo_model = YOLO("yolov8n.pt")

# Text sentiment analysis
def analyze_text_sentiment(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    sentiment_score = torch.softmax(outputs.logits, dim=1).tolist()[0]

    input_ids = gpt2_tokenizer.encode(text, return_tensors="pt")
    generated = gpt2_model.generate(input_ids, max_length=50, num_return_sequences=1)
    gpt2_output = gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)

    return {"bert_sentiment": sentiment_score, "gpt2_response": gpt2_output}

# Visual sentiment analysis
def analyze_visual_sentiment(image_path):
    results = yolo_model(image_path)[0]
    class_ids = results.boxes.cls.tolist()
    names_dict = results.names
    objects_detected = [names_dict[int(i)] for i in class_ids]

    face_analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
    emotions = [face["dominant_emotion"] for face in face_analysis] if isinstance(face_analysis, list) else [face_analysis["dominant_emotion"]]

    return {"objects_detected": objects_detected, "emotions_detected": emotions}

# Multimodal fusion
def multimodal_sentiment_fusion(text_result, image_result):
    return {
        "text_sentiment": text_result["bert_sentiment"],
        "generated_text": text_result["gpt2_response"],
        "objects_detected": image_result["objects_detected"],
        "facial_emotions": image_result["emotions_detected"]
    }

# Streamlit UI
st.title("🧠 Multimodal Sentiment Intelligence Platform")

text_input = st.text_area("Enter your text")

uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png"])

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        text_result = analyze_text_sentiment(text_input)

        if uploaded_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_image.read())
                image_path = tmp_file.name
                image_result = analyze_visual_sentiment(image_path)
                result = multimodal_sentiment_fusion(text_result, image_result)
        else:
            result = text_result

    st.subheader("🔍 Sentiment Analysis Result")
    st.json(result)
