import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# -----------------------
# Load Model (Cached)
# -----------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    return Sequential([base_model, GlobalMaxPooling2D()])

model = load_model()

# -----------------------
# Load Stored Embeddings
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "embeddings.pkl"), "rb") as f:
    feature_list = pickle.load(f)

with open(os.path.join(BASE_DIR, "filenames.pkl"), "rb") as f:
    filenames = pickle.load(f)

# Clean filenames to ensure they point inside ajio_images
filenames = [os.path.join("ajio_images", os.path.basename(f)) for f in filenames]

# -----------------------
# Feature Extraction
# -----------------------
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# -----------------------
# Recommend Function
# -----------------------
def recommend(features, feature_list, n_neighbors=5):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return distances, indices

# -----------------------
# Streamlit UI
# -----------------------
st.subheader(" Fashion Recommendation System ")
st.markdown("Upload a fashion product and get **similar style recommendations** ")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Save uploaded image
    uploaded_path = os.path.join(BASE_DIR, "uploaded.jpg")
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_path, caption="Uploaded Image", width=180)

    # Extract features
    features = extract_features(uploaded_path, model)

    # Get Recommendations
    distances, indices = recommend(features, feature_list, n_neighbors=6)

    st.subheader("Recommended Similar Products")

    cols = st.columns(5)
    for i, col in enumerate(cols):
        img_path = os.path.join(BASE_DIR, filenames[indices[0][i+1]])  # skip first (self)
        if os.path.exists(img_path):
            col.image(img_path, use_container_width=True)
        else:
            col.warning(f"Missing: {img_path}")

    # -----------------------
    # "Accuracy" Metric (Cosine Similarity)
    # -----------------------
   # cosine_similarities = []
    #for idx in indices[0][1:]:
    #    rec_features = feature_list[idx]
     #   cos_sim = np.dot(features, rec_features) / (norm(features) * norm(rec_features))
     #   cosine_similarities.append(cos_sim)

    #avg_similarity = np.mean(cosine_similarities)
   # st.metric("Recommendation Accuracy (Avg. Cosine Similarity)", f"{avg_similarity:.2f}")
