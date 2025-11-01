# app.py
import os
import pickle
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# Torch + Transformers
import torch
from transformers import CLIPProcessor, CLIPModel

# Keras for ResNet (you can use timm/efficientnet in place)
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image as kimage

st.set_page_config(page_title="Fashion Recommender", layout="centered")
st.title("👗 Fashion Recommender")

# -------------------------
# Load CLIP (cached)
# -------------------------
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

clip_model, clip_processor = load_clip()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# -------------------------
# Load ResNet (cached)
# -------------------------
@st.cache_resource
def load_resnet():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    model = tf.keras.Sequential([base, GlobalMaxPooling2D()])
    return model

resnet_model = load_resnet()

# -------------------------
# Color histogram (RGB)
# -------------------------
def color_histogram(img_pil: Image.Image, bins_per_channel=32):
    img = img_pil.convert("RGB")
    arr = np.array(img.resize((224,224)))
    # compute histogram per channel
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch].ravel(), bins=bins_per_channel, range=(0,255))
        hist.append(h.astype(float))
    hist = np.concatenate(hist)
    # normalize
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist  # length = 3*bins_per_channel

# -------------------------
# CLIP image embedding
# -------------------------
def extract_clip_embedding_from_pil(img_pil: Image.Image):
    inputs = clip_processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)  #  (1, dim)
    emb = emb.cpu().numpy().reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

# -------------------------
# ResNet embedding
# -------------------------
def extract_resnet_embedding_from_pil(img_pil: Image.Image):
    img = img_pil.resize((224,224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet_preprocess(x)
    feat = resnet_model.predict(x, verbose=0).flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-10)
    return feat

# -------------------------
# Combined feature
# -------------------------
def combined_embedding_from_pil(img_pil: Image.Image,
                                w_clip=1.0, w_res=0.8, w_color=0.4):
    clip_v = extract_clip_embedding_from_pil(img_pil)
    res_v = extract_resnet_embedding_from_pil(img_pil)
    color_v = color_histogram(img_pil, bins_per_channel=32)
    # match dimensions: clip_v (~512), res_v (~2048), color_v (96)
    # scale weights and concat
    vec = np.concatenate([w_clip * clip_v, w_res * res_v, w_color * color_v])
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec

# -------------------------
# Load dataset CSV(s)
# -------------------------
csv_files = ["dress.csv"]  # update with your CSV(s) list or multiple files
dfs = []
for file in csv_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="latin1")
        df["source_file"] = file
        dfs.append(df)
    else:
        st.warning(f"{file} not found — skipping")

if not dfs:
    st.error("No CSVs found. Place dress.csv (or dataset files) in the app folder.")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)
# Ensure necessary columns exist
if "image_url" not in df_all.columns or "product_url" not in df_all.columns or "title" not in df_all.columns:
    st.error("CSV must have 'image_url', 'product_url', and 'title' columns.")
    st.stop()

st.success(f"Loaded {len(df_all)} rows. Preparing / loading embeddings...")

# -------------------------
# Build / load embeddings cache
# -------------------------
EMB_PATH = "hybrid_embeddings.pkl"
VALID_CSV = "valid_dress_products.csv"

# recompute? Provide button to recompute
recompute = st.button("🔁 Recompute all embeddings (force)")

if recompute and os.path.exists(EMB_PATH):
    os.remove(EMB_PATH)
    if os.path.exists(VALID_CSV):
        os.remove(VALID_CSV)

if os.path.exists(EMB_PATH):
    st.info("Loading cached embeddings...")
    embeddings = pickle.load(open(EMB_PATH, "rb"))
    valid_df = pd.read_csv(VALID_CSV)
else:
    st.info("Computing embeddings for dataset images (may take time)...")
    embeddings = []
    valid_rows = []
    pbar = st.progress(0)
    total = len(df_all)
    i = 0
    for _, row in tqdm(df_all.iterrows(), total=total):
        url = row["image_url"]
        try:
            resp = requests.get(url, timeout=8)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            vec = combined_embedding_from_pil(img, w_clip=1.0, w_res=0.8, w_color=0.4)
            embeddings.append(vec)
            valid_rows.append(row)
        except Exception as e:
            # skip bad image
            print(f"Skipping {url}: {e}")
        i += 1
        if total>0:
            pbar.progress(min(100, int(i/total*100)))
    valid_df = pd.DataFrame(valid_rows)
    pickle.dump(embeddings, open(EMB_PATH, "wb"))
    valid_df.to_csv(VALID_CSV, index=False)
    st.success(f"Saved {len(embeddings)} embeddings.")

# Convert list to numpy for faster similarity
if isinstance(embeddings, list):
    embeddings = np.vstack(embeddings) if len(embeddings)>0 else np.zeros((0,1))

# -------------------------
# Upload query image and recommend
# -------------------------
st.write("Upload a query image to get top-5 dress recommendations:")
uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])

# sliders to tweak weights
st.sidebar.header("Embedding weights")
w_clip = st.sidebar.slider("CLIP weight", 0.0, 2.0, 1.0, 0.05)
w_res = st.sidebar.slider("ResNet weight", 0.0, 2.0, 0.8, 0.05)
w_color = st.sidebar.slider("Color weight", 0.0, 1.0, 0.4, 0.05)

if uploaded:
    qimg = Image.open(uploaded).convert("RGB")
    st.image(qimg, caption="Query Image", use_column_width=True)

    # compute query embedding using same pipeline but with updated weights
    def combined_query(img_pil):
        clip_v = extract_clip_embedding_from_pil(img_pil)
        res_v = extract_resnet_embedding_from_pil(img_pil)
        color_v = color_histogram(img_pil, bins_per_channel=32)
        vec = np.concatenate([w_clip * clip_v, w_res * res_v, w_color * color_v])
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    qvec = combined_query(qimg)

    if embeddings.shape[0] == 0:
        st.error("No embeddings available to compare.")
    else:
        sims = cosine_similarity([qvec], embeddings)[0]
        top_k = int(st.sidebar.slider("Number of results", 1, 10, 5))
        idxs = np.argsort(sims)[-top_k:][::-1]

        st.subheader("Top recommendations")
        for rank, idx in enumerate(idxs, start=1):
            score = sims[idx]
            row = valid_df.iloc[idx]
            img_url = row["image_url"]
            product_url = row["product_url"]
            title = row.get("title", "")
            source = row.get("source_file", "")

            # show small card
            st.markdown(f"### {rank}. {title}  —  **{score*100:.2f}%**")
            st.markdown(f"Source: `{source}`")
            st.markdown(f'<a href="{product_url}" target="_blank"><img src="{img_url}" width="300"></a>', unsafe_allow_html=True)
            st.markdown(f"[Open product link]({product_url})")
            st.markdown("---")

# End
