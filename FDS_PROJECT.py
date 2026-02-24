import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import math

# Page setup
st.set_page_config(page_title="Image Compression using SVD", layout="wide")

# -------------------- FUNCTIONS --------------------

def svd_compress(img_array, k):
    """
    Compress image using top-k singular values
    """
    U, S, Vt = np.linalg.svd(img_array, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return np.clip(compressed, 0, 255), S

def mse(original, compressed):
    """
    Mean Squared Error
    """
    return np.mean((original - compressed) ** 2)

def psnr(original, compressed):
    """
    Peak Signal-to-Noise Ratio
    """
    error = mse(original, compressed)
    if error == 0:
        return 100
    return 20 * math.log10(255 / math.sqrt(error))

# -------------------- UI --------------------

st.title("🖼️ Image Compression using SVD")

uploaded_file = st.file_uploader(
    "Upload an Image (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Load and convert image
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image, dtype=np.float64)

    m, n = img_array.shape

    # Slider for k
    k = st.slider(
        "Number of Singular Values (k)",
        min_value=1,
        max_value=min(m, n),
        value=50
    )

    # Compression
    compressed_img, singular_values = svd_compress(img_array, k)

    # -------------------- DISPLAY --------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Compressed Image (k = {k})")
        st.image(compressed_img.astype(np.uint8), use_container_width=True)

    # -------------------- METRICS --------------------

    mse_val = mse(img_array, compressed_img)
    psnr_val = psnr(img_array, compressed_img)

    st.markdown("### 📊 Compression Metrics")
    st.write(f"**MSE:** {mse_val:.2f}")
    st.write(f"**PSNR:** {psnr_val:.2f} dB")

    # -------------------- PLOT --------------------

    fig, ax = plt.subplots()
    ax.plot(singular_values)
    ax.set_title("Singular Values Distribution")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    # -------------------- DOWNLOAD --------------------

    buffer = io.BytesIO()
    Image.fromarray(compressed_img.astype(np.uint8)).save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Compressed Image",
        data=buffer.getvalue(),
        file_name="compressed_image.png",
        mime="image/png"
    )
