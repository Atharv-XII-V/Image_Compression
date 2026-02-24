import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Image Compression using SVD",
    layout="wide"
)

# ---------------- HERO HEADER ----------------
st.markdown("""
# 🖼️ Image Compression using SVD
### *Low-rank image compression with visual & quantitative analysis*

Supports **Grayscale & RGB images** • Adjustable compression • Metrics & plots • Download output
""")

st.divider()

# ---------------- FUNCTIONS ----------------

def svd_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return np.clip(compressed, 0, 255), S

def mse(original, compressed):
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    return np.mean((original - compressed) ** 2)

def psnr(original, compressed):
    err = mse(original, compressed)
    if err == 0:
        return 100
    return 20 * math.log10(255 / math.sqrt(err))

# ---------------- SIDEBAR (CONTROL PANEL) ----------------
st.sidebar.markdown("## 🎛️ Control Panel")
st.sidebar.markdown("Adjust compression strength")

uploaded_file = st.sidebar.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image, dtype=np.float64)

    is_color = (len(img_array.shape) == 3)
    h, w = img_array.shape[:2]
    max_k = min(h, w)

    k = st.sidebar.slider(
        "Number of Singular Values (k)",
        min_value=1,
        max_value=max_k,
        value=min(50, max_k),
        help="Higher k → better quality, lower compression"
    )

    # ---------------- COMPRESSION ----------------
    if is_color:
        R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        R_c, S_r = svd_channel(R, k)
        G_c, S_g = svd_channel(G, k)
        B_c, S_b = svd_channel(B, k)
        compressed_img = np.dstack((R_c, G_c, B_c))
        singular_values = (S_r + S_g + S_b) / 3
    else:
        compressed_img, singular_values = svd_channel(img_array, k)

    compressed_img = compressed_img.astype(np.uint8)

    # ---------------- METRICS ----------------
    mse_val = mse(img_array, compressed_img)
    psnr_val = psnr(img_array, compressed_img)

    original_size = h * w * (3 if is_color else 1)
    compressed_size = k * (h + w + 1) * (3 if is_color else 1)
    compression_ratio = original_size / compressed_size

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["🖼️ Images", "📊 Metrics", "📈 Analysis"])

    # -------- TAB 1: IMAGE COMPARISON --------
    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader(f"Compressed Image (k = {k})")
            st.image(compressed_img, use_container_width=True)

    # -------- TAB 2: METRICS --------
    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{mse_val:.2f}", help="Lower is better")
        c2.metric("PSNR (dB)", f"{psnr_val:.2f}", help="Higher is better")
        c3.metric("Compression Ratio", f"{compression_ratio:.2f} : 1")

    # -------- TAB 3: ANALYSIS --------
    with tab3:

        # Singular Values
        st.subheader("🔍 Singular Value Distribution")
        fig, ax = plt.subplots()
        ax.plot(singular_values)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Singular Values (Energy Content)")
        st.pyplot(fig)

        # k values
        k_values = np.linspace(1, max_k, num=10, dtype=int)

        # k vs PSNR
        st.subheader("📈 k vs PSNR")
        psnr_values = []
        for ki in k_values:
            if is_color:
                Rk, _ = svd_channel(img_array[:, :, 0], ki)
                Gk, _ = svd_channel(img_array[:, :, 1], ki)
                Bk, _ = svd_channel(img_array[:, :, 2], ki)
                reconstructed = np.dstack((Rk, Gk, Bk))
            else:
                reconstructed, _ = svd_channel(img_array, ki)
            psnr_values.append(psnr(img_array, reconstructed))

        fig2, ax2 = plt.subplots()
        ax2.plot(k_values, psnr_values, marker='o')
        ax2.set_xlabel("k")
        ax2.set_ylabel("PSNR (dB)")
        ax2.grid(True)
        st.pyplot(fig2)

        # k vs Compression Ratio
        st.subheader("📉 k vs Compression Ratio")
        compression_ratios = [original_size / (ki * (h + w + 1) * (3 if is_color else 1)) for ki in k_values]
        fig3, ax3 = plt.subplots()
        ax3.plot(k_values, compression_ratios, marker='o')
        ax3.set_xlabel("k")
        ax3.set_ylabel("Compression Ratio")
        ax3.grid(True)
        st.pyplot(fig3)

        # Energy Retained
        st.subheader("📊 Energy Retained (%) vs k")
        total_energy = np.sum(singular_values ** 2)
        energy_retained = [(np.sum(singular_values[:ki] ** 2) / total_energy) * 100 for ki in k_values]
        fig4, ax4 = plt.subplots()
        ax4.plot(k_values, energy_retained, marker='o')
        ax4.set_xlabel("k")
        ax4.set_ylabel("Energy Retained (%)")
        ax4.grid(True)
        st.pyplot(fig4)

    # ---------------- EXPORT ----------------
    st.divider()
    st.subheader("⬇️ Export Result")

    buffer = io.BytesIO()
    Image.fromarray(compressed_img).save(buffer, format="PNG")

    st.download_button(
        label="Download Compressed Image",
        data=buffer.getvalue(),
        file_name="compressed_image.png",
        mime="image/png",
        use_container_width=True
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption("Developed using Python, NumPy & Streamlit • Image Compression using SVD")
