
import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

# Page configuration
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# Load pretrained model only once
@st.cache_resource
def load_model():
    model_name = "Salesforce/blip-image-captioning-base"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return processor, model, device


# Generate caption
def generate_caption(image, processor, model, device):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = inputs.to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# Application interface
st.title("🖼️ AI Image Caption Generator")

st.write(
    "Upload an image and let Artificial Intelligence "
    "describe what it sees."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("✨ Generate Caption", type="primary"):

        try:
            with st.spinner(
                "Loading AI model and generating caption..."
            ):
                processor, model, device = load_model()

                caption = generate_caption(
                    image,
                    processor,
                    model,
                    device
                )

            st.success("Caption generated successfully!")

            st.subheader("Generated Caption")
            st.write(caption)

            st.caption(f"Model running on: {device.upper()}")

        except Exception as error:
            st.error(f"Something went wrong: {error}")

else:
    st.info("Please upload an image to get started.")

st.divider()
st.caption("Powered by Python, PyTorch, Hugging Face and BLIP")
# python -m streamlit run app1.py