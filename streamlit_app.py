import os
import sys
import subprocess
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO
import torch
import tempfile
from PIL import Image
import gc
from concurrent.futures import ThreadPoolExecutor
import psutil

# ====================== INITIAL SETUP ======================
def check_and_install_packages():
    """Self-healing package installation that works with Streamlit Cloud"""
    required_packages = {
        "streamlit": "1.28.0",
        "torch": "2.0.1+cpu",
        "transformers": "4.33.3",
        "ultralytics": "8.0.196",
        "opencv-python-headless": "4.8.0.76",
        "pillow": "10.0.1",
        "numpy": "1.24.4",
        "markdown-it-py": "2.2.0",
        "mdurl": "0.1.2",
        "rich": "13.7.0",
        "psutil": "5.9.5",
        "tqdm": "4.66.1"
    }
    
    for pkg, version in required_packages.items():
        try:
            if pkg == "torch":
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    f"{pkg}=={version}", "--index-url", "https://download.pytorch.org/whl/cpu"
                ])
            else:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    f"{pkg}=={version}"
                ])
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install {pkg}: {e}")
            return False
    return True

# Must be the first Streamlit command
st.set_page_config(
    page_title="🧠 Multimodal Sentiment Intelligence",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Multimodal Sentiment Analysis Platform"
    }
)

# Check and install packages
if not check_and_install_packages():
    st.error("Critical dependency installation failed. Please check logs.")
    st.stop()

# ====================== MODEL LOADING ======================
@st.cache_resource(show_spinner="Loading NLP models...")
def load_models():
    """Load and cache all ML models"""
    try:
        # Text models
        bert_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        bert_model = BertForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            torch_dtype=torch.float16
        )
        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Vision model
        yolo_model = YOLO("yolov8n.pt").to('cpu')
        
        return {
            "bert": (bert_tokenizer, bert_model),
            "gpt2": (gpt2_tokenizer, gpt2_model),
            "yolo": yolo_model
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# ====================== ANALYSIS FUNCTIONS ======================
def analyze_text_sentiment(text, tokenizer, model):
    """Analyze text sentiment using BERT"""
    try:
        text = text[:512]  # Limit input size
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
        return probs.index(max(probs)) + 1, probs
    except Exception as e:
        raise Exception(f"Sentiment analysis failed: {str(e)}")

def generate_response(text, tokenizer, model):
    """Generate response using GPT-2"""
    try:
        inputs = tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise Exception(f"Response generation failed: {str(e)}")

def detect_objects(image, model):
    """Detect objects in image using YOLO"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name)
            if not results or not results[0].boxes:
                return []
            return list(set(results[0].names[int(cls)] for cls in results[0].boxes.cls))
    except Exception as e:
        raise Exception(f"Object detection failed: {str(e)}")
    finally:
        gc.collect()

# ====================== MAIN APPLICATION ======================
def main():
    st.title("🧠 Multimodal Sentiment Intelligence Platform")
    st.markdown("""
    Analyze text sentiment and image content using:
    - 🤖 BERT for text sentiment
    - 🎨 YOLOv8 for object detection
    - 💬 GPT-2 for response generation
    """)

    # Load models
    try:
        models = load_models()
        bert_tokenizer, bert_model = models["bert"]
        gpt2_tokenizer, gpt2_model = models["gpt2"]
        yolo_model = models["yolo"]
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return

    # Input form
    with st.form("analysis_form"):
        text_input = st.text_area("📝 Enter text to analyze:", height=150)
        image_file = st.file_uploader("🖼️ Upload image (optional):", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("🚀 Analyze")

    if submitted:
        if not text_input.strip():
            st.warning("Please enter some text")
            return

        # Memory check
        if psutil.virtual_memory().percent > 85:
            st.error("High memory usage - please try smaller inputs")
            return

        # Analysis
        with st.spinner("Analyzing..."):
            try:
                # Parallel execution
                with ThreadPoolExecutor() as executor:
                    # Submit tasks
                    sentiment_future = executor.submit(
                        analyze_text_sentiment, text_input, bert_tokenizer, bert_model
                    )
                    response_future = executor.submit(
                        generate_response, text_input, gpt2_tokenizer, gpt2_model
                    )
                    
                    # Get results
                    rating, scores = sentiment_future.result()
                    response = response_future.result()
                    
                    # Display results
                    st.subheader("📊 Text Analysis Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("⭐ Sentiment Rating", f"{rating}/5")
                        st.write("📊 Sentiment Scores:", scores)
                    with col2:
                        st.write("💬 Generated Response:")
                        st.info(response)
                    
                    # Image analysis if provided
                    if image_file:
                        st.subheader("🖼️ Image Analysis")
                        image = Image.open(image_file)
                        st.image(image, use_column_width=True)
                        objects = detect_objects(image, yolo_model)
                        st.write("🔍 Detected Objects:", ", ".join(objects) if objects else "No objects detected")
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    main()
