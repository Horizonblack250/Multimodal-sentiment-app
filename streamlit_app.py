import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO
import torch
import tempfile
from PIL import Image
import gc
from concurrent.futures import ThreadPoolExecutor
import psutil

# ---------------------- Initial Config (Must be First) ----------------------
st.set_page_config(
    page_title="🧠 Multimodal Sentiment Intelligence",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Multimodal Sentiment Analysis Platform"
    }
)

# ---------------------- Memory Management ----------------------
def check_memory():
    mem = psutil.virtual_memory()
    if mem.percent > 85:
        st.warning(f"High memory usage: {mem.percent}%. Analysis may be unstable.")
        return False
    return True

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

# ---------------------- Cached Model Loaders ----------------------
@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert():
    try:
        tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        model = BertForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            torch_dtype=torch.float16  # Use half precision to save memory
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load BERT model: {str(e)}")
        raise

@st.cache_resource(show_spinner="Loading GPT-2 model...")
def load_gpt2():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load GPT-2 model: {str(e)}")
        raise

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo():
    try:
        model = YOLO("yolov8n.pt").to('cpu')  # Force CPU usage
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {str(e)}")
        raise

# ---------------------- Analysis Functions ----------------------
def analyze_text_sentiment(text, tokenizer, model):
    try:
        text = text[:512]  # Trim long input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment_probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
        predicted_rating = sentiment_probs.index(max(sentiment_probs)) + 1
        return predicted_rating, sentiment_probs
    except Exception as e:
        raise Exception(f"Text sentiment analysis failed: {str(e)}")

def generate_gpt2_response(text, tokenizer, model):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception as e:
        raise Exception(f"Response generation failed: {str(e)}")

def analyze_image_objects(image, model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            results = model(tmp_file.name)
            
            if not results or not results[0].boxes or not results[0].boxes.cls.numel():
                return []
            
            names = results[0].names
            detected = [names[int(cls)] for cls in results[0].boxes.cls]
            return list(set(detected))
    except Exception as e:
        raise Exception(f"Image analysis failed: {str(e)}")
    finally:
        cleanup_memory()

# ---------------------- Main Application ----------------------
def main():
    st.title("🧠 Multimodal Sentiment Intelligence Platform")
    st.markdown("""
    Analyze text sentiment and image content in real-time using:
    - 🤖 BERT for text sentiment analysis
    - 🎨 YOLOv8 for object detection
    - 💬 GPT-2 for response generation
    """)

    # Initialize models
    try:
        bert_tokenizer, bert_model = load_bert()
        gpt2_tokenizer, gpt2_model = load_gpt2()
        yolo_model = load_yolo()
    except Exception as e:
        st.error(f"Critical error: Could not load models. {str(e)}")
        st.stop()

    with st.form("input_form"):
        text = st.text_area("📄 Enter your review/message:", 
                          placeholder="Type your text here...",
                          height=150)
        image_file = st.file_uploader("🖼️ Upload an image (optional):", 
                                    type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=False)
        submit = st.form_submit_button("🚀 Analyze")

    if submit:
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return

        if not check_memory():
            st.error("Insufficient memory to perform analysis. Please try with smaller inputs.")
            return

        st.subheader("📊 Analysis Results")
        progress_bar = st.progress(0)
        
        try:
            # Text analysis in parallel
            with ThreadPoolExecutor() as executor:
                # Submit tasks
                future_sentiment = executor.submit(
                    analyze_text_sentiment, text, bert_tokenizer, bert_model
                )
                future_response = executor.submit(
                    generate_gpt2_response, text, gpt2_tokenizer, gpt2_model
                )
                
                # Get results
                rating, sentiment_scores = future_sentiment.result()
                gpt2_response = future_response.result()
                
                progress_bar.progress(50)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⭐ Predicted Sentiment Rating", f"{rating}/5")
                    st.write("📈 Sentiment Distribution:", sentiment_scores)
                with col2:
                    st.write("💬 Generated Response:")
                    st.info(gpt2_response)
                
                # Image analysis if provided
                if image_file:
                    progress_bar.progress(75)
                    try:
                        image = Image.open(image_file)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        objects = analyze_image_objects(image, yolo_model)
                        st.write("🔍 Detected Objects:", ", ".join(objects) if objects else "No objects detected")
                    except Exception as e:
                        st.error(str(e))
                
                progress_bar.progress(100)
                st.success("Analysis complete!")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
        finally:
            cleanup_memory()
            progress_bar.empty()

if __name__ == "__main__":
    main()
