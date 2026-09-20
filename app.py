import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import wikipedia # --- NEW: The Wikipedia Library ---

# Setup the Web Page
st.set_page_config(page_title="Image Classifier AI", page_icon="🤖")
st.title("General Image Classifier 🖼️")
st.write("Upload any picture, and the AI will try to guess what it is!")

# Load the Pre-Trained AI (MobileNetV2)
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

# Create a File Uploader for the User
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the image if uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display Image Metadata
    st.subheader("📊 Image Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        file_ext = uploaded_file.name.split('.')[-1].upper()
        st.metric("Format", image.format or file_ext)
    with col2:
        st.metric("Dimensions", f"{image.size[0]} x {image.size[1]}")
    with col3:
        file_size_kb = uploaded_file.size / 1024
        st.metric("File Size", f"{file_size_kb:.2f} KB")
    st.divider()

    st.image(image, caption='Your Uploaded Image', use_container_width=True)
    st.write("🧠 AI is thinking...")

    # Convert the image to RGB (removes any transparency channels)
    image = image.convert('RGB')

    # Resize and convert image to numbers for the AI
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Get the AI's prediction
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    # --- NEW: Get Information about the Top Prediction ---
    top_prediction_label = decoded_predictions[0][1].replace('_', ' ').title()
    
    st.success("Done!")
    st.subheader(f"💡 What is a {top_prediction_label}?")
    
    # Try to search Wikipedia for the top guess
    try:
        with st.spinner("Fetching information..."):
            # Get a 3-sentence summary from Wikipedia
            wiki_summary = wikipedia.summary(top_prediction_label, sentences=3)
            st.write(wiki_summary)
            
            # Provide a link for them to read more
            wiki_url = f"https://en.wikipedia.org/wiki/{top_prediction_label.replace(' ', '_')}"
            st.markdown(f"**[Read more about it on Wikipedia]({wiki_url})**")
    except wikipedia.exceptions.DisambiguationError:
        st.write("This is a very broad topic, but it looks like a variation of this object!")
    except wikipedia.exceptions.PageError:
        st.write("The AI recognizes this object, but we couldn't find a specific Wikipedia page for it.")
    except Exception:
        st.write("Could not load additional information right now.")
        
    st.divider()
    
    # Display the rest of the confidence scores
    st.subheader("🔍 AI Confidence Breakdown:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i + 1}. **{label.replace('_', ' ').title()}** ({score * 100:.2f}% confidence)")
        st.progress(float(score))