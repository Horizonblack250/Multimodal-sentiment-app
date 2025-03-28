import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO
import torch
import tempfile
from PIL import Image

# ---------------------- Cached Model Loaders ----------------------

@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

# Load all models
bert_tokenizer, bert_model = load_bert()
gpt2_tokenizer, gpt2_model = load_gpt2()
yolo_model = load_yolo()

# ---------------------- Analysis Functions ----------------------

def analyze_text_sentiment(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bert_model(**inputs)
    sentiment_probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
    predicted_rating = sentiment_probs.index(max(sentiment_probs)) + 1

    input_ids = gpt2_tokenizer.encode(text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    generated = gpt2_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        temperature=0.7,
        do_sample=True,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    response = gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)

    return predicted_rating, sentiment_probs, response

def analyze_image_objects(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        results = yolo_model(tmp_file.name)
        names = results[0].names if results else []
        detected = [names[int(cls)] for cls in results[0].boxes.cls]
    return list(set(detected))

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="🧠 Multimodal Sentiment Intelligence", layout="centered")
st.title("🧠 Multimodal Sentiment Intelligence Platform")

with st.form("input_form"):
    text = st.text_area("📄 Enter your review / message:")
    image_file = st.file_uploader("🖼️ Upload an image (optional):", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Analyze")

if submit:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.subheader("📊 Analysis Results")

        # Text sentiment + GPT-2 response
        rating, sentiment_scores, gpt2_response = analyze_text_sentiment(text)
        st.write(f"⭐ **Predicted Sentiment Rating**: {rating} / 5")
        st.write("📈 **Sentiment Distribution**:", sentiment_scores)
        st.write("🗣️ **GPT-2 Generated Response:**", gpt2_response)

        # Image object detection
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            objects = analyze_image_objects(image)
            st.write("🔍 **Objects Detected in Image**:", objects)
