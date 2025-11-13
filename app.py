# app.py
import os
import pickle
from io import BytesIO
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import CLIPProcessor, CLIPModel
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image as kimage

# ------------------------- #
# PAGE CONFIG & STYLING
# ------------------------- #
st.set_page_config(page_title="Fashion Recommendation System", layout="centered")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #0D0D0D;
            color: #E0F7FA;
        }
        .stButton>button {
            background: linear-gradient(90deg, #00BCD4, #4FC3F7);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 18px;
            font-weight: 600;
            box-shadow: 0 0 15px rgba(79,195,247,0.4);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            box-shadow: 0 0 25px rgba(79,195,247,0.6);
            transform: scale(1.03);
        }
        .upload-box {
            border: 2px dashed #4FC3F7;
            border-radius: 15px;
            text-align: center;
            padding: 25px;
            color: #E0F7FA;
            box-shadow: 0 0 15px rgba(79,195,247,0.2);
            transition: all 0.3s ease;
        }
        .upload-box:hover {
            box-shadow: 0 0 25px rgba(79,195,247,0.4);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------- #
# TITLE
# ------------------------- #
st.markdown("<h1 style='text-align:center;color:#4FC3F7;'>👗 Fashion Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#B2EBF2;'>Find your next outfit with AI-powered recommendations.</p>", unsafe_allow_html=True)

# ------------------------- #
# LOAD MODELS
# ------------------------- #
@st.cache_resource
def load_clip():
    print("[INFO] Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    print("[INFO] CLIP model loaded successfully.")
    return model, processor

clip_model, clip_processor = load_clip()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
print(f"[INFO] Using device: {device}")

@st.cache_resource
def load_resnet():
    print("[INFO] Loading ResNet50 model...")
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    model = tf.keras.Sequential([base, GlobalMaxPooling2D()])
    print("[INFO] ResNet50 loaded successfully.")
    return model

resnet_model = load_resnet()

# ------------------------- #
# FEATURE EXTRACTION
# ------------------------- #
def color_histogram(img_pil: Image.Image, bins_per_channel=32):
    img = img_pil.convert("RGB")
    arr = np.array(img.resize((224,224)))
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch].ravel(), bins=bins_per_channel, range=(0,255))
        hist.append(h.astype(float))
    hist = np.concatenate(hist)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist

def extract_clip_embedding_from_pil(img_pil: Image.Image):
    inputs = clip_processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb.cpu().numpy().reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb

def extract_resnet_embedding_from_pil(img_pil: Image.Image):
    img = img_pil.resize((224,224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet_preprocess(x)
    feat = resnet_model.predict(x, verbose=0).flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-10)
    return feat

def combined_embedding_from_pil(img_pil: Image.Image, w_clip=1.0, w_res=0.8, w_color=0.4):
    clip_v = extract_clip_embedding_from_pil(img_pil)
    res_v = extract_resnet_embedding_from_pil(img_pil)
    color_v = color_histogram(img_pil, bins_per_channel=32)
    vec = np.concatenate([w_clip * clip_v, w_res * res_v, w_color * color_v])
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec

# ------------------------- #
# DEDUP FUNCTION
# ------------------------- #
def remove_near_duplicates(embeddings, df, threshold=0.985, batch_size=500):
    """
    Fast deduplication for medium datasets (~2k–5k).
    """
    if isinstance(embeddings, list):
        embeddings = np.vstack(embeddings)
    n = len(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    keep = np.ones(n, dtype=bool)
    processed = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = np.dot(embeddings[start:end], embeddings.T)
        np.fill_diagonal(sims, 0)
        dup_mask = (sims > threshold)
        for i in range(start, end):
            if not keep[i]:
                continue
            dups = np.where(dup_mask[i - start])[0]
            dups = dups[dups > i]
            keep[dups] = False
        processed += (end - start)
        if processed % 500 == 0 or processed == n:
            print(f"[INFO] Processed {processed}/{n} embeddings...")

    new_embeddings = embeddings[keep]
    new_df = df.iloc[np.where(keep)[0]].reset_index(drop=True)
    removed = n - np.sum(keep)
    print(f"[INFO] Removed {removed} duplicates (threshold={threshold})")
    return new_embeddings, new_df

# ------------------------- #
# LOAD DATASET
# ------------------------- #
csv_files = ["dress.csv", "dress2.csv"]
dfs = []
print("[INFO] Loading dataset files...")
for file in csv_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="latin1", on_bad_lines='skip')
        df["source_file"] = file
        dfs.append(df)
        print(f"[OK] Loaded {len(df)} rows from {file}")
    else:
        print(f"[WARN] {file} not found, skipping...")

if not dfs:
    st.error("No dataset found. Please add dress.csv or similar.")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)
print(f"[INFO] Total {len(df_all)} rows combined.")

# ------------------------- #
# EMBEDDINGS CACHE + DEDUP (ONE-TIME)
# ------------------------- #
EMB_PATH = "hybrid_embeddings.pkl"
VALID_CSV = "valid_dress_products.csv"
DEDUP_PATH = "hybrid_embeddings_dedup.pkl"
CLEAN_CSV = "valid_dress_products_dedup.csv"

recompute = st.button("🔁 Recompute all embeddings (force)")

if recompute:
    for f in [EMB_PATH, VALID_CSV, DEDUP_PATH, CLEAN_CSV]:
        if os.path.exists(f):
            os.remove(f)
    st.experimental_rerun()

with st.spinner("🧠 Loading embeddings..."):
    if os.path.exists(DEDUP_PATH) and os.path.exists(CLEAN_CSV):
        print("[INFO] Using deduplicated cache.")
        embeddings = pickle.load(open(DEDUP_PATH, "rb"))
        valid_df = pd.read_csv(CLEAN_CSV)

    elif os.path.exists(EMB_PATH) and os.path.exists(VALID_CSV):
        print("[INFO] Loading raw cached embeddings (first-time dedup)...")
        embeddings = pickle.load(open(EMB_PATH, "rb"))
        valid_df = pd.read_csv(VALID_CSV)

        embeddings, valid_df = remove_near_duplicates(embeddings, valid_df, threshold=0.985)
        pickle.dump(embeddings, open(DEDUP_PATH, "wb"))
        valid_df.to_csv(CLEAN_CSV, index=False)
        print("[INFO] Deduplication done and cached permanently.")

        os.remove(EMB_PATH)
        os.remove(VALID_CSV)
        print("[INFO] Old raw caches deleted. Now using deduped data only.")

    else:
        print("[INFO] No cache found — computing all embeddings...")
        embeddings, valid_rows = [], []
        for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
            url = row["image_url"]
            try:
                resp = requests.get(url, timeout=8)
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                vec = combined_embedding_from_pil(img)
                embeddings.append(vec)
                valid_rows.append(row)
            except Exception as e:
                print(f"[WARN] Skipping image: {url} — {e}")
                continue

        valid_df = pd.DataFrame(valid_rows)
        pickle.dump(embeddings, open(EMB_PATH, "wb"))
        valid_df.to_csv(VALID_CSV, index=False)
        print(f"[INFO] Saved {len(embeddings)} raw embeddings successfully.")

if isinstance(embeddings, list):
    embeddings = np.vstack(embeddings)

# ------------------------- #
# IMAGE UPLOAD + RESULTS
# ------------------------- #
st.markdown("<div class='upload-box'>⬆️ Upload a dress image to get top 5 AI recommendations.</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.sidebar.header("Embedding Weights")
w_clip = st.sidebar.slider("CLIP Weight", 0.0, 2.0, 1.0, 0.05)
w_res = st.sidebar.slider("ResNet Weight", 0.0, 2.0, 0.8, 0.05)
w_color = st.sidebar.slider("Color Weight", 0.0, 1.0, 0.4, 0.05)

if uploaded:
    qimg = Image.open(uploaded).convert("RGB")
    st.image(qimg, caption="Uploaded Image", use_column_width=True)

    with st.spinner("✨ Generating recommendations... Please wait."):
        clip_v = extract_clip_embedding_from_pil(qimg)
        res_v = extract_resnet_embedding_from_pil(qimg)
        color_v = color_histogram(qimg)
        qvec = np.concatenate([w_clip * clip_v, w_res * res_v, w_color * color_v])
        qvec = qvec / (np.linalg.norm(qvec) + 1e-10)

        sims = cosine_similarity([qvec], embeddings)[0]
        top_k = int(st.sidebar.slider("Number of results", 1, 10, 5))
        idxs = np.argsort(sims)[-top_k:][::-1]

    st.markdown("<h3 style='color:#4FC3F7;text-align:center;margin-top:30px;'>✨ Top Fashion Recommendations ✨</h3>", unsafe_allow_html=True)
    cols = st.columns(2)
    for rank, idx in enumerate(idxs, start=1):
        score = sims[idx]
        row = valid_df.iloc[idx]
        img_url = row["image_url"]
        product_url = row["product_url"]
        title = row.get("title", "Fashion Item")

        col = cols[(rank - 1) % 2]
        with col:
            st.markdown(f"""
                <div style="text-align:center;padding:15px;border-radius:15px;
                background-color:#121212;box-shadow:0 0 15px rgba(79,195,247,0.15);margin-bottom:25px;">
                    <img src="{img_url}" width="260" style="border-radius:12px;margin-bottom:10px;">
                    <h4 style="color:#E0F7FA;margin:5px 0;">{rank}. {title}</h4>
                    <p style="color:#4FC3F7;margin:4px 0;">Similarity: <b>{score*100:.2f}%</b></p>
                    <a href="{product_url}" target="_blank">
                        <button style="background:linear-gradient(90deg,#00BCD4,#4FC3F7);
                            color:white;border:none;border-radius:10px;padding:8px 16px;font-weight:600;
                            cursor:pointer;box-shadow:0 0 10px rgba(79,195,247,0.3);
                            transition:all 0.3s ease;">🔗 View Product</button>
                    </a>
                </div>
            """, unsafe_allow_html=True)
