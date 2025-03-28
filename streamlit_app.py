import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO
import torch
import tempfile
from PIL import Image
import gc

# ====================== STREAMLIT CONFIG ======================
st.set_page_config(
    page_title="🧠 Multimodal Sentiment Intelligence",
    layout="centered"
)

st.title("🧠 Multimodal Sentiment Intelligence")

# ====================== MODEL LOADING ======================
@st.cache_resource(show_spinner="🔄 Loading models...")
def load_models():
    """Load BERT, GPT-2, and YOLO models with device fallback"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Text models
    bert_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    bert_model = BertForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    ).to(device)

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Vision model
    yolo_model = YOLO("yolov8n.pt")  # Runs on CPU by default

    return {
        "bert": (bert_tokenizer, bert_model),
        "gpt2": (gpt2_tokenizer, gpt2_model),
        "yolo": yolo_model,
        "device": device
    }

# ====================== MAIN FUNCTION ======================
def main():
    # Load models
    try:
        models = load_models()
        bert_tokenizer, bert_model = models["bert"]
        gpt2_tokenizer, gpt2_model = models["gpt2"]
        yolo_model = models["yolo"]
        device = models["device"]
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        return

    # Input section
    with st.form("input_form"):
        text = st.text_area("📝 Enter text:", height=150)
        image_file = st.file_uploader("🖼️ Upload image (optional):", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("🚀 Analyze")

    if submitted:
        if not text.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("🔍 Analyzing..."):
            try:
                # ====================== TEXT ANALYSIS ======================
                inputs = bert_tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True).to(device)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
                rating = probs.index(max(probs)) + 1

                input_ids = gpt2_tokenizer.encode(text[:512], return_tensors="pt").to(device)
                with torch.no_grad():
                    generated = gpt2_model.generate(
                        input_ids,
                        max_length=100,
                        pad_token_id=gpt2_tokenizer.eos_token_id
                    )
                response = gpt2_tokenizer.decode(generated[0], skip_special_tokens=True)

                # ====================== DISPLAY TEXT RESULTS ======================
                st.subheader("📊 Sentiment & Response")
                st.write(f"⭐ **Sentiment Rating**: {rating}/5")
                st.write("💬 **GPT-2 Response**:")
                st.markdown(f"> {response}")

                # ====================== IMAGE ANALYSIS ======================
                if image_file:
                    image = Image.open(image_file).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        results = yolo_model(tmp.name)
                        objects = list(set(results[0].names[int(x)] for x in results[0].boxes.cls)) if results[0].boxes else []
                        st.subheader("🖼️ Detected Objects")
                        st.write(", ".join(objects) if objects else "No objects detected.")

                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"🚨 Error during analysis: {str(e)}")
            finally:
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

