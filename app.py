"""
Deteksi Kemiripan Wajah dengan PCA/SVD (Eigenfaces)
====================================================
Implementasi sesuai tugas:
  - Preprocessing: grayscale, resize, normalisasi, flatten
  - Centering data (mean face)
  - PCA via SVD
  - Proyeksi ke ruang PCA
  - Cosine Similarity & Euclidean Distance
  - Threshold & keputusan mirip / tidak mirip
  - Visualisasi eigenfaces & mean face
  - Face detection via Haar Cascade
"""

import streamlit as st
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tempfile
import os

# ─────────────────────────────────────────────
#  Konfigurasi halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Kemiripan Wajah – PCA/SVD",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CSS kustom
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header utama */
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        color: #ffffff;
    }
    .hero h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
        color: #a78bfa;
    }
    .hero p {
        font-size: 1rem;
        color: #c4b5fd;
        margin: 0;
        line-height: 1.6;
    }

    /* Kartu metrik */
    .metric-card {
        background: #1e1b4b;
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        color: #e0e7ff;
    }
    .metric-card .label {
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #a78bfa;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #6366f1;
        margin-top: 0.2rem;
    }

    /* Badge hasil */
    .badge-match {
        background: #14532d;
        border: 1px solid #16a34a;
        color: #4ade80;
        border-radius: 50px;
        padding: 0.5rem 1.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .badge-nomatch {
        background: #450a0a;
        border: 1px solid #dc2626;
        color: #f87171;
        border-radius: 50px;
        padding: 0.5rem 1.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Section label */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        color: #6366f1;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #312e81;
        margin: 2rem 0;
    }

    /* Info box */
    .info-box {
        background: #0f172a;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    .info-box code {
        background: #1e293b;
        color: #a78bfa;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Streamlit override */
    .stButton > button {
        background: linear-gradient(135deg, #4338ca, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }

    div[data-testid="stExpander"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Konstanta
# ─────────────────────────────────────────────
IMG_SIZE = (100, 100)
N_PIX    = IMG_SIZE[0] * IMG_SIZE[1]   # 10 000

# ─────────────────────────────────────────────
#  Fungsi utilitas
# ─────────────────────────────────────────────

@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def preprocess_image(file_bytes: bytes, detector) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Baca bytes gambar → deteksi wajah → grayscale → resize → normalisasi → flatten.
    Return: (vector 10000,), (crop wajah grayscale 100x100) atau (None, None) jika gagal.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # Fallback: gunakan seluruh gambar jika wajah tidak terdeteksi
        face_crop = gray
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray[y:y+h, x:x+w]

    face_resized    = cv2.resize(face_crop, IMG_SIZE)
    face_normalized = face_resized / 255.0
    return face_normalized.flatten(), face_resized


def build_pca(X: np.ndarray, n_components: int):
    """
    Sesuai PDF:
      1. Hitung mean face  X̄
      2. Centering: Xc = X - X̄
      3. SVD pada Xc → PCA
    Return: pca_model, mean_face, Xc, X_pca
    """
    mean_face = X.mean(axis=0)          # shape (10000,)
    Xc        = X - mean_face           # centering

    n_comp    = min(n_components, len(X) - 1)
    pca       = PCA(n_components=n_comp)
    X_pca     = pca.fit_transform(Xc)   # Proyeksi ke ruang PCA

    return pca, mean_face, Xc, X_pca


def compare(z1: np.ndarray, z2: np.ndarray):
    """Hitung Cosine Similarity dan Euclidean Distance antara dua vektor PCA."""
    cos  = float(cosine_similarity(z1.reshape(1, -1), z2.reshape(1, -1))[0][0])
    euc  = float(np.linalg.norm(z1 - z2))
    return cos, euc


def fig_to_array(fig) -> np.ndarray:
    """Konversi matplotlib figure ke numpy array RGB."""
    fig.canvas.draw()
    buf   = fig.canvas.tostring_rgb()
    w, h  = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 Deteksi Kemiripan Wajah</h1>
  <p>Implementasi <strong>PCA / SVD (Eigenfaces)</strong> untuk membandingkan dua wajah.<br>
  Setiap gambar direduksi dari <strong>10.000 piksel → k dimensi PCA</strong>,
  lalu kemiripan dihitung via Cosine Similarity dan Euclidean Distance.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR – Parameter
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter PCA")
    n_components = st.slider("Jumlah komponen PCA (k)", 2, 50, 20,
                             help="Seberapa banyak eigenface yang dipakai")
    cos_thresh   = st.slider("Threshold Cosine Similarity", 0.50, 0.99, 0.75,
                             help="Di atas nilai ini → MIRIP")
    euc_thresh   = st.slider("Threshold Euclidean Distance", 5.0, 100.0, 30.0,
                             help="Di bawah nilai ini → MIRIP")

    st.markdown("---")
    st.markdown("### 📖 Ringkasan Alur")
    st.markdown("""
<div class="info-box">
1. Upload ≥2 foto referensi (database)<br>
2. Upload 2 foto yang akan dibandingkan<br>
3. Preprocessing: <code>grayscale → resize → normalize → flatten</code><br>
4. <code>Xc = X − mean_face</code> (centering)<br>
5. <code>SVD(Xc) → eigenfaces</code><br>
6. <code>Z = Xc · Vk</code> (proyeksi)<br>
7. Hitung <code>cosine_sim</code> & <code>euclidean_dist</code><br>
8. Bandingkan dengan threshold
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TAB UTAMA
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏋️ Latih Model (Database)",
    "🔍 Bandingkan Dua Wajah",
    "🌐 Kenali dari Database",
])

detector = load_detector()

# ══════════════════════════════════════════════
#  TAB 1 – Dataset & Training PCA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Train Data</p>', unsafe_allow_html=True)
    st.subheader("Upload Foto Referensi (Database Wajah)")

    st.markdown("""
    <div class="info-box">
    Upload <strong>minimal 3 foto</strong> sebagai data latih PCA.
    Bisa foto satu orang atau banyak orang — sistem akan mempelajari
    <em>variasi</em> wajah dan membangun <strong>eigenfaces</strong>.
    </div>
    """, unsafe_allow_html=True)

    db_files = st.file_uploader(
        "Upload foto database (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="db_upload",
    )

    if db_files and len(db_files) >= 2:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Baca & preprocessing semua gambar database ──
        vectors, crops, names = [], [], []
        fail_count = 0
        for f in db_files:
            vec, crop = preprocess_image(f.read(), detector)
            if vec is not None:
                vectors.append(vec)
                crops.append(crop)
                names.append(f.name)
            else:
                fail_count += 1

        if fail_count:
            st.warning(f"{fail_count} gambar tidak dapat dibaca dan dilewati.")

        if len(vectors) < 2:
            st.error("Butuh minimal 2 gambar yang valid.")
        else:
            X          = np.array(vectors)         # (m, 10000)
            pca, mean_face, Xc, X_pca = build_pca(X, n_components)

            # Simpan ke session_state agar bisa dipakai tab lain
            st.session_state["pca"]       = pca
            st.session_state["mean_face"] = mean_face
            st.session_state["X_pca"]     = X_pca
            st.session_state["X"]         = X
            st.session_state["names"]     = names
            st.session_state["crops"]     = crops

            # ── Ringkasan matriks ──
            k_actual = pca.n_components_
            explained = float(np.sum(pca.explained_variance_ratio_)) * 100

            col1, col2, col3, col4 = st.columns(4)
            for col, label, val, sub in [
                (col1, "Jumlah Gambar (m)", str(len(X)), f"matriks X ∈ ℝ^{len(X)}×{N_PIX}"),
                (col2, "Fitur per Gambar (n)", f"{N_PIX:,}", "100×100 piksel"),
                (col3, "Komponen PCA (k)", str(k_actual), "eigenfaces dipakai"),
                (col4, "Explained Variance", f"{explained:.1f}%", "informasi terjaga"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="label">{label}</div>
                      <div class="value">{val}</div>
                      <div class="sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            # ── Mean Face ──
            st.markdown("#### 🎭 Mean Face (Wajah Rata-rata)")
            st.markdown("""
            <div class="info-box">
            <code>X̄ = mean(X, axis=0)</code> — rata-rata seluruh piksel dari semua wajah latih.<br>
            Setiap wajah kemudian dikurangi mean face ini sebelum PCA: <code>Xc = X − X̄</code>
            </div>
            """, unsafe_allow_html=True)

            fig_mean, ax_mean = plt.subplots(figsize=(2.5, 2.5), facecolor="#0f172a")
            ax_mean.imshow(mean_face.reshape(IMG_SIZE), cmap="gray")
            ax_mean.axis("off")
            ax_mean.set_title("Mean Face", color="#a78bfa",
                               fontsize=9, fontfamily="monospace")
            st.pyplot(fig_mean, use_container_width=False)
            plt.close(fig_mean)

            # ── Eigenfaces ──
            st.markdown("#### 🔬 Eigenfaces (Komponen Utama PCA)")
            st.markdown("""
            <div class="info-box">
            Kolom-kolom dari matriks <code>V</code> hasil SVD divisualisasikan sebagai <em>eigenfaces</em>.<br>
            Setiap eigenface menangkap satu arah variasi terbesar dari data wajah.
            </div>
            """, unsafe_allow_html=True)

            n_show    = min(k_actual, 10)
            n_cols_ef = 5
            n_rows_ef = (n_show + n_cols_ef - 1) // n_cols_ef

            fig_ef, axes_ef = plt.subplots(
                n_rows_ef, n_cols_ef,
                figsize=(n_cols_ef * 2, n_rows_ef * 2.2),
                facecolor="#0f172a"
            )
            axes_flat = axes_ef.flatten() if hasattr(axes_ef, 'flatten') else [axes_ef]
            for i, ax in enumerate(axes_flat):
                if i < n_show:
                    eigenface = pca.components_[i].reshape(IMG_SIZE)
                    ax.imshow(eigenface, cmap="RdPu")
                    ev_ratio  = pca.explained_variance_ratio_[i] * 100
                    ax.set_title(f"EF-{i+1}\n{ev_ratio:.1f}%",
                                 color="#c4b5fd", fontsize=7, fontfamily="monospace")
                ax.axis("off")
            fig_ef.tight_layout(pad=0.4)
            st.pyplot(fig_ef, use_container_width=True)
            plt.close(fig_ef)

            # ── Pratinjau gambar database ──
            st.markdown("#### 🗂️ Preview Database Wajah (setelah preprocessing)")
            n_prev = min(len(crops), 8)
            cols_prev = st.columns(n_prev)
            for i, (col, crop, name) in enumerate(zip(cols_prev, crops[:n_prev], names[:n_prev])):
                with col:
                    st.image(crop, caption=name[:15], use_container_width=True,
                             clamp=True)

            st.success(f"✅ Model PCA berhasil dibangun dari {len(X)} gambar dengan {k_actual} komponen.")

    elif db_files and len(db_files) < 2:
        st.info("Upload minimal 2 foto untuk membangun model PCA.")
    else:
        st.info("👆 Upload foto database terlebih dahulu.")


# ══════════════════════════════════════════════
#  TAB 2 – Bandingkan Dua Wajah
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Cosine & Euclidean</p>',
                unsafe_allow_html=True)
    st.subheader("Bandingkan Dua Gambar Wajah")

    if "pca" not in st.session_state:
        st.warning("⚠️ Latih model PCA di tab **🏋️ Latih Model** terlebih dahulu.")
    else:
        pca_m     = st.session_state["pca"]
        mean_face = st.session_state["mean_face"]

        c1, c2 = st.columns(2)
        with c1:
            f1 = st.file_uploader("📸 Wajah ke-1", type=["jpg","jpeg","png"], key="face1")
        with c2:
            f2 = st.file_uploader("📸 Wajah ke-2", type=["jpg","jpeg","png"], key="face2")

        if f1 and f2:
            vec1, crop1 = preprocess_image(f1.read(), detector)
            vec2, crop2 = preprocess_image(f2.read(), detector)

            if vec1 is None or vec2 is None:
                st.error("Gagal membaca salah satu gambar.")
            else:
                # Centering & proyeksi sesuai PDF: Z = Xc · Vk
                z1 = pca_m.transform((vec1 - mean_face).reshape(1, -1))[0]
                z2 = pca_m.transform((vec2 - mean_face).reshape(1, -1))[0]

                cos_sim, euc_dist = compare(z1, z2)

                cos_match = cos_sim  >= cos_thresh
                euc_match = euc_dist <= euc_thresh

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # Tampilkan dua wajah
                pc1, pc2, pc3 = st.columns([2, 1, 2])
                with pc1:
                    st.image(crop1, caption="Wajah ke-1 (preprocessed)",
                             use_container_width=True, clamp=True)
                with pc2:
                    st.markdown("<br><br><br><h2 style='text-align:center;color:#6366f1'>↔</h2>",
                                unsafe_allow_html=True)
                with pc3:
                    st.image(crop2, caption="Wajah ke-2 (preprocessed)",
                             use_container_width=True, clamp=True)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                # ── Metrik ──
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="label">Cosine Similarity</div>
                      <div class="value">{cos_sim:.4f}</div>
                      <div class="sub">Threshold: ≥ {cos_thresh:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="label">Euclidean Distance</div>
                      <div class="value">{euc_dist:.2f}</div>
                      <div class="sub">Threshold: ≤ {euc_thresh:.2f}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Badge hasil
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    badge_cos = "badge-match" if cos_match else "badge-nomatch"
                    icon_cos  = "✅ MIRIP" if cos_match else "❌ TIDAK MIRIP"
                    st.markdown(f"""
                    <p style="color:#818cf8;font-size:0.75rem;margin-bottom:4px;">
                      Cosine Similarity</p>
                    <span class="{badge_cos}">{icon_cos}</span>""",
                                unsafe_allow_html=True)
                with bcol2:
                    badge_euc = "badge-match" if euc_match else "badge-nomatch"
                    icon_euc  = "✅ MIRIP" if euc_match else "❌ TIDAK MIRIP"
                    st.markdown(f"""
                    <p style="color:#818cf8;font-size:0.75rem;margin-bottom:4px;">
                      Euclidean Distance</p>
                    <span class="{badge_euc}">{icon_euc}</span>""",
                                unsafe_allow_html=True)

                # ── Visualisasi vektor PCA ──
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                with st.expander("📊 Lihat Representasi Vektor PCA", expanded=False):
                    k_vis = min(len(z1), 20)
                    fig_vec, ax_vec = plt.subplots(figsize=(10, 3), facecolor="#0f172a")
                    x_idx = np.arange(k_vis)
                    ax_vec.bar(x_idx - 0.2, z1[:k_vis], 0.4,
                               color="#7c3aed", alpha=0.85, label="Wajah ke-1")
                    ax_vec.bar(x_idx + 0.2, z2[:k_vis], 0.4,
                               color="#db2777", alpha=0.85, label="Wajah ke-2")
                    ax_vec.set_facecolor("#0f172a")
                    ax_vec.tick_params(colors="#94a3b8")
                    ax_vec.spines[:].set_color("#1e293b")
                    ax_vec.set_xlabel("Komponen PCA ke-i", color="#94a3b8",
                                      fontsize=9)
                    ax_vec.set_ylabel("Nilai Proyeksi", color="#94a3b8", fontsize=9)
                    ax_vec.set_title(
                        f"Representasi z₁ dan z₂ di Ruang PCA "
                        f"(menampilkan {k_vis} dari {len(z1)} komponen)",
                        color="#c4b5fd", fontsize=9)
                    ax_vec.legend(facecolor="#1e293b", labelcolor="white",
                                  fontsize=8)
                    st.pyplot(fig_vec, use_container_width=True)
                    plt.close(fig_vec)

                    # Penjelasan matematis
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>Formula yang digunakan:</strong><br>
                    Centering: <code>xc = x − X̄</code><br>
                    Proyeksi:  <code>z = PCA.transform(xc)</code>  →  <code>Z = Xc · Vk</code><br>
                    Cosine:    <code>sim(z₁,z₂) = (z₁·z₂) / (‖z₁‖‖z₂‖) = {cos_sim:.4f}</code><br>
                    Euclidean: <code>d(z₁,z₂) = ‖z₁−z₂‖ = {euc_dist:.4f}</code>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 3 – Identifikasi dari Database
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">recognize_face()</p>',
                unsafe_allow_html=True)
    st.subheader("Kenali Wajah dari Database")
    st.markdown("""
    <div class="info-box">
    Upload satu foto, sistem akan mencari wajah <strong>paling mirip</strong>
    dari seluruh database yang sudah dilatih — menggunakan Cosine Similarity.
    </div>
    """, unsafe_allow_html=True)

    if "pca" not in st.session_state:
        st.warning("⚠️ Latih model PCA di tab **🏋️ Latih Model** terlebih dahulu.")
    else:
        pca_m     = st.session_state["pca"]
        mean_face = st.session_state["mean_face"]
        X_pca     = st.session_state["X_pca"]
        db_names  = st.session_state["names"]
        db_crops  = st.session_state["crops"]

        test_file = st.file_uploader("Upload foto wajah baru",
                                     type=["jpg","jpeg","png"], key="recog")

        if test_file:
            vec_test, crop_test = preprocess_image(test_file.read(), detector)

            if vec_test is None:
                st.error("Gagal membaca gambar.")
            else:
                # Proyeksi wajah uji ke ruang PCA
                z_test = pca_m.transform((vec_test - mean_face).reshape(1, -1))[0]

                # Hitung cosine similarity ke semua wajah di database
                sims = cosine_similarity(z_test.reshape(1, -1), X_pca)[0]

                best_idx    = int(np.argmax(sims))
                best_score  = float(sims[best_idx])
                best_name   = db_names[best_idx]
                best_crop   = db_crops[best_idx]

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    st.image(crop_test, caption="Wajah Input", use_container_width=True)
                with rc2:
                    st.image(best_crop, caption=f"Paling Mirip: {best_name}",
                             use_container_width=True)

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                col_s, col_r = st.columns(2)
                with col_s:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="label">Best Cosine Similarity</div>
                      <div class="value">{best_score:.4f}</div>
                      <div class="sub">{best_score*100:.1f}% kemiripan</div>
                    </div>""", unsafe_allow_html=True)
                with col_r:
                    is_match = best_score >= cos_thresh
                    badge    = "badge-match" if is_match else "badge-nomatch"
                    verdict  = f"✅ COCOK — {best_name}" if is_match else "❌ TIDAK DIKENAL"
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="label">Hasil Identifikasi</div>
                      <br>
                      <span class="{badge}">{verdict}</span>
                    </div>""", unsafe_allow_html=True)

                # Ranking semua similarity
                with st.expander("📋 Ranking Similarity ke Seluruh Database"):
                    sorted_idx = np.argsort(sims)[::-1]
                    rows = []
                    for rank, idx in enumerate(sorted_idx, 1):
                        rows.append({
                            "Rank": rank,
                            "Nama File": db_names[idx],
                            "Cosine Similarity": f"{sims[idx]:.4f}",
                            "Status": "✅ Mirip" if sims[idx] >= cos_thresh else "❌ Tidak Mirip"
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;color:#374151;font-size:0.8rem;font-family:'IBM Plex Mono',monospace;">
  Implementasi PCA/SVD · Eigenfaces · Deteksi Kemiripan Wajah
</p>
""", unsafe_allow_html=True)
