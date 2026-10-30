import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------------------------------------------------
# App UI & Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Crop Health AI", page_icon="🌱", layout="centered")
st.title("🌱 Crop Health & Disease Detection")
st.markdown("Upload or capture a photo of your crop to get an instant diagnosis and treatment plan powered by Gemma 3.")

# Crop Selection
crop_options = ["Rice", "Wheat", "Maize", "Barley"]
selected_crop = st.selectbox("Select Crop:", crop_options)

# Image Input (Camera or Upload)
st.subheader("1. Provide a Crop Image")
image = None
input_method = st.radio("Choose input method:", ("Upload File", "Use Camera"), horizontal=True)

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_file = st.camera_input("Capture Image")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

if image:
    st.image(image, caption="Provided Image", use_column_width=True)

# ---------------------------------------------------------
# Model Loading (Cached to prevent reloading on UI changes)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Gemma 3 Model into memory (this takes a minute)...")
def load_model():
    # Using the instruction-tuned version of Gemma 3 (best for local VS Code)
    model_id = "google/gemma-3-4b-it"
   
    # Load processor and model
    # using float16 and auto device mapping to manage memory efficiently
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return processor, model

# ---------------------------------------------------------
# Processing & Inference
# ---------------------------------------------------------
if image:
    if st.button("Analyze Crop Health", type="primary"):
        try:
            processor, model = load_model()
           
            # Construct the Gemma 3 prompt
            system_prompt = f"""
            You are an agricultural expert. Analyze the provided image of a {selected_crop} plant and diagnose any diseases.
            Provide your response clearly structured with these three exact headings:
           
            ### Disease Diagnosis
            (Describe symptoms and identify the disease, or state "No disease detected")
           
            ### Recovery Methods
            (2-3 specific actionable steps)
           
            ### Fertilizer & Treatment
            (Specific fertilizer, pesticide, or natural treatment recommendations)
            """

            # Gemma 3 expects a specific multimodal message format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]
           
            # Process inputs using the chat template
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
           
            # Generate output
            with st.spinner("Gemma 3 is analyzing the crop..."):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.2,
                    do_sample=True
                )
               
            # Decode and clean up output
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
           
            # Isolate the model's response from the prompt
            if "model\n" in generated_text:
                final_answer = generated_text.split("model\n")[-1].strip()
            else:
                final_answer = generated_text
           
            # Display Results
            st.success("Analysis Complete!")
            st.markdown(final_answer)

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.info("Tip: Ensure you have accepted the Gemma 3 license on Hugging Face and logged in via the terminal (`huggingface-cli login`).")
