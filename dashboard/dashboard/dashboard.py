import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import base64

# =============================================================
# CONFIG
# =============================================================
st.set_page_config(
    page_title="Learn Guard Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# LOAD DATA (Simulasi - Nanti diganti dengan kode asli Anda)
# =============================================================
@st.cache_data
def load_data():
    # Pastikan path ini sesuai dengan lokasi file Anda saat dijalankan
    try:
        df_weekly = pd.read_csv('data/processed/wrangled/df_master_weekly_clean.csv')
        df_weekly = df_weekly[df_weekly["week"].isin([1, 2, 3, 4])].copy()
        df_weekly["risk_group"] = df_weekly["risk_label"].map({
            0: "Pass/Distinction",
            1: "Fail/Withdrawn"
        })

        df_final = pd.read_csv("data/processed/df_final.csv")
        df_final["risk_group"] = df_final["risk_label"].map({
            0: "Pass/Distinction",
            1: "Fail/Withdrawn"
        })
        return df_weekly, df_final
    except FileNotFoundError:
        st.warning("File data tidak ditemukan. Menggunakan data dummy untuk demo.")
        # Data dummy agar dashboard tetap bisa dilihat
        df_weekly = pd.DataFrame({
            "week": [1, 2, 3, 4],
            "count": [22963, 23168, 23822, 22135]
        })
        df_final = pd.DataFrame({
            "risk_label": [0, 0, 1, 0, 1],
            "score": [80, 90, 40, 75, 30]
        })
        df_final["risk_group"] = df_final["risk_label"].map({
            0: "Pass/Distinction",
            1: "Fail/Withdrawn"
        })
        return df_weekly, df_final

df_weekly, df_final = load_data()

palette = {"Pass/Distinction": "#2ecc71", "Fail/Withdrawn": "#e74c3c"}
# =============================================================
# SIDEBAR MENU
# =============================================================
st.markdown(
    """
    <style>
    /* Target the sidebar container */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    /* Optional: Style text to ensure visibility on dark background */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stTitle,
    [data-testid="stSidebar"] .stRadio {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("🧭 Navigasi")
    # background_color method removed as it doesn't exist
    
    menu_options = ["Stat Data Bersih", "Pertanyaan Bisnis", "Model MLM"]
    choice = st.radio("Pilih Menu:", menu_options)

# =============================================================
# HALAMAN: STAT DATA BERSIH (HOME)
# =============================================================
if choice == "Stat Data Bersih":
    st.title("📊 Stat Data Bersih & Insight")
    st.markdown("Ringkasan kualitas data dan statistik persiapan sebelum analisis lebih lanjut.")
    st.markdown("---")

    # --- BAGIAN 1: STATISTIK UTAMA (METRICS) ---
    st.subheader("Ringkasan Statistik")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status Data", "Bersih", help="0 baris dihapus, tanpa negative weeks.")
    with col2:
        st.metric("Total Siswa (df_final)", "27,795", help="Data awal siswa.")
    with col3:
        st.metric("Siswa Analisis (df_pb2)", "20,818", help="Setelah filter data lengkap.")
    with col4:
        st.metric("Rasio Class Imbalance", "58:42", help="Pass vs Fail (Moderat).")

    st.markdown("---")

    # --- BAGIAN 2: VISUALISASI GAMBAR (FRAME HITAM & BACKGROUND PUTIH) ---
    st.subheader("Visualisasi Data")
    
    img_folder = "data/img" 
    images = [
        "01_data_quality.png",
        "02_distribution mingguan.png",
        "03_distribusi_risk_data.png",
        "04_data_lost.png",
        "05_distribusi_korelasi.png"
    ]

    # Styling CSS untuk membuat efek frame hitam dan background putih
    # Kita gunakan container HTML untuk memastikan background putih di sekitar gambar
    css_frame = """
    <style>
    .img-frame {
        border: 3px solid black;
        background-color: white;
        padding: 15px;
        margin-bottom: 30px;
        border-radius: 8px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .img-frame img {
        max-width: 100%;
        height: auto;
        display: block;
    }
    .insight-box {
        background-color: #172C43;
        margin-bottom: 40px;
        padding: 15px;
        border-left: 5px solid #2ecc71;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.95em;
    }
    .insight-title {
        font-weight: bold;
        color: #2ecc71;
        margin-bottom: 5px;
        display: block;
    }
    </style>
    """
    st.markdown(css_frame, unsafe_allow_html=True)

    for img_name in images:
        img_path = os.path.join(img_folder, img_name)
        img_title = img_name.replace(".png", "").replace("_", " ").title()
        insight_text = "" # Variabel untuk menampung insight

        if os.path.exists(img_path):
            # Membaca gambar sebagai base64
            with open(img_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode()

            # Menentukan Insight berdasarkan nama file
            if "quality" in img_name.lower():
                insight_text = "Data bersih tanpa penghapusan baris."
                label = "Kualitas Data"
            elif "mingguan" in img_name.lower():
                insight_text = "Distribusi mingguan seimbang (~25% per minggu)."
                label = "Distribusi Waktu"
            elif "risk" in img_name.lower():
                insight_text = "Rasio 58:42 (Pass:Fail). Moderat, disarankan pakai class_weight."
                label = "Imbalance Ratio"
            elif "lost" in img_name.lower():
                insight_text = "25.1% data hilang di df_pb2 (siswa tanpa tugas)."
                label = "Data Loss"
            elif "korelasi" in img_name.lower():
                insight_text = "Analisis hanya berlaku untuk siswa yang submit tugas."
                label = "Cakupan Data"
            else:
                insight_text = "Visualisasi menunjukkan pola umum dari dataset."
                label = "Insight Umum"

            # Render Judul
            st.markdown(f"#### {img_title}")
            
            # Render Gambar dengan Frame
            st.markdown(f"""
            <div class="img-frame">
                <img src="data:image/png;base64,{base64_img}">
            </div>
            """, unsafe_allow_html=True)
            
            # Render Insight Box
            st.markdown(f"""
            <div class="insight-box">
                <span class="insight-title">{label}</span>
                {insight_text}
            </div>
            """, unsafe_allow_html=True)
            # Tambahkan spasi vertikal antar item (menggunakan margin-bottom pada insight-box di CSS)
        else:
            st.error(f"Gambar tidak ditemukan: `{img_path}`")

    # --- BAGIAN 3: INSIGHT DETAIL (TEKS LEBIH PANJANG) ---
    st.markdown("---")
    st.subheader("Insight Mendalam")
    
    with st.expander("Baca Detail Insight Persiapan Data", expanded=False):
        st.markdown("""
        **1. Kualitas Data: Sudah Bersih Sempurna**
        *   **Removed rows:** `0`
        *   **Implikasi:** Tidak ada *negative weeks* yang lolos ke tahap analisis. Pipeline pembersihan data berfungsi optimal.
        *   **Status:** Data siap dianalisis tanpa *preprocessing* tambahan.

        **2. Distribusi Mingguan: Sangat Seimbang**
        Distribusi data antar minggu hampir merata (~25% per minggu), menjamin analisis tren tidak bias.
        *   **Minggu 3:** Puncak aktivitas (kemungkinan deadline).
        *   **Minggu 4:** Penurunan aktivitas (konsisten dengan temuan BQ3).

        **3. Kondisi Class Imbalance: Moderat (58:42)**
        Rasio kelas Pass (58.4%) vs Fail (41.6%) termasuk *imbalanced* namun tidak ekstrem.
        *   **Handling:** Cukup gunakan `class_weight="balanced"`. Tidak perlu SMOTE.
        *   **Metrik:** Gunakan **F1-score** dan **ROC-AUC**. *Accuracy* saja tidak cukup (Baseline: 58.4%).

        **4. Kehilangan Data pada df_pb2**
        Terjadi pengurangan signifikan dari 27,795 siswa menjadi 20,818 siswa (hilang ~25.1%).
        *   **Penyebab:** Siswa yang *Withdrawn* sangat awal tidak punya data `days_diff` atau `score`.
        *   **Dampak:** Analisis korelasi `days_diff` vs `score` **hanya merepresentasikan** siswa aktif. Siswa risiko tertinggi (tidak submit) tidak tercakup di sini.

        **5. Validasi Skala Data**
        *   **Shape Identik:** `df_early` dan `df_full` sama-sama **92,088 × 17**.
        *   **Konfirmasi:** Filter perbaikan berhasil. Data hanya mencakup **Minggu 1–4**.
        """)

# =============================================================
# HALAMAN: PERTANYAAN BISNIS
# =============================================================
elif choice == "Pertanyaan Bisnis":
    st.title("Pertanyaan Bisnis & Analisis")
    st.markdown("Pilih pertanyaan bisnis untuk melihat analisis mendalam dan interaktif.")
    st.markdown("---")

    # --- SUB-MENU PILIHAN PERTANYAAN ---
        # --- SUB-MENU PILIHAN PERTANYAAN ---
    bq_options = {
        "Pertanyaan Bisnis 1: Rata-rata Klik Mingguan (Pass vs Fail)": "bq1",
        "Pertanyaan Bisnis 2: Korelasi Waktu Submit vs Skor": "bq2",
        "Pertanyaan Bisnis 3: Penurunan Aktivitas Demografi": "bq3"
    }
    
    selected_bq = st.selectbox("Pilih Pertanyaan Bisnis:", list(bq_options.keys()))
    bq_key = bq_options[selected_bq]

    # =============================================================
    # LOGIKA BQ1 (Sudah Dikonversi)
    # =============================================================
    if bq_key == "bq1":
        st.header("Pertanyaan Bisnis 1: Rata-rata Klik Mingguan per Grup Risiko")
        st.markdown(
            "_Berapa rata-rata jumlah klik mingguan pada 4 minggu pertama yang "
            "membedakan siswa **Pass/Distinction** vs **Fail/Withdrawn**?_"
        )

        # Filter interaktif
        selected_weeks = st.multiselect(
            "Pilih Minggu untuk Analisis:",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            key="bq1_weeks_selector"
        )

        if not selected_weeks:
            st.warning("Silakan pilih minimal satu minggu.")
        else:
            df_bq1 = df_weekly[df_weekly["week"].isin(selected_weeks)].copy()

            # Agregasi per student
            df_student = df_bq1.groupby(
                ["id_student", "code_module", "code_presentation", "risk_group"]
            )["total_clicks"].mean().reset_index()

            # Weekly mean
            df_weekly_filtered = df_bq1.groupby(
                ["id_student", "code_module", "code_presentation", "risk_group", "week"]
            )["total_clicks"].sum().reset_index()

            weekly_mean = df_weekly_filtered.groupby(
                ["risk_group", "week"]
            )["total_clicks"].mean().reset_index()

            # Stats
            stats_bq1 = df_student.groupby("risk_group")["total_clicks"].agg(
                mean="mean", median="median", std="std", count="count"
            ).round(2)

            st.dataframe(stats_bq1, use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Tren Klik per Minggu")
                fig, ax = plt.subplots(figsize=(5, 4))
                for group, grp_df in weekly_mean.groupby("risk_group"):
                    ax.plot(
                        grp_df["week"], grp_df["total_clicks"],
                        marker="o", label=group,
                        color=palette[group], linewidth=2.5
                    )
                ax.set_xlabel("Minggu")
                ax.set_ylabel("Rata-rata Total Clicks")
                ax.set_xticks(selected_weeks)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Bar Chart per Minggu")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.barplot(
                    data=df_weekly_filtered,
                    x="week", y="total_clicks",
                    hue="risk_group",
                    palette=palette,
                    estimator="mean",
                    errorbar="sd",
                    ax=ax
                )
                ax.set_xlabel("Minggu")
                ax.set_ylabel("Rata-rata Total Clicks")
                ax.legend(title="Grup", fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col3:
                st.subheader("Distribusi per Siswa")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.boxplot(
                    data=df_student,
                    x="risk_group", y="total_clicks",
                    palette=palette,
                    showfliers=False,
                    ax=ax
                )
                ax.set_xlabel("")
                ax.set_ylabel("Avg Clicks")
                ax.tick_params(axis="x", rotation=10)
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Insight BQ1
            pass_mean = stats_bq1.loc["Pass/Distinction", "mean"]
            fail_mean = stats_bq1.loc["Fail/Withdrawn", "mean"]
            selisih   = pass_mean - fail_mean
            pct       = selisih / fail_mean * 100 if fail_mean != 0 else 0

            st.info(
                f"**Insight:** Siswa Pass/Distinction rata-rata mengklik "
                f"**{pass_mean:.1f} kali/minggu**, sedangkan Fail/Withdrawn hanya "
                f"**{fail_mean:.1f} kali/minggu** — selisih **{selisih:.1f} klik "
                f"({pct:.1f}% lebih tinggi)** secara konsisten di semua minggu. "
                f"Kesenjangan ini sekitar **20-30 klik** lebih banyak untuk kelompok lulus."
            )


    # =============================================================
    # LOGIKA BQ2 (Korelasi Waktu Submit vs Skor)
    # =============================================================
    elif bq_key == "bq2":
        st.header("Pertanyaan Bisnis 2: Korelasi Waktu Submit vs Skor Akhir")
        st.markdown(
            "_Bagaimana korelasi antara selisih waktu pengumpulan tugas pertama "
            "(**days_diff**) dengan skor akhir yang diperoleh siswa?_"
        )

        # Persiapan Data
        # Pastikan kolom ada, jika tidak tampilkan pesan error
        required_cols = ["days_diff", "score", "risk_group"]
        if not all(col in df_final.columns for col in required_cols):
            st.error(f"Data tidak lengkap. Kolom yang dibutuhkan: {required_cols}")
            st.stop()

        df_bq2 = df_final[["days_diff", "score", "risk_group"]].dropna()
        
        # Logika tambahan: Menentukan status terlambat
        df_bq2["is_late"] = (df_bq2["days_diff"] > 0).astype(int)
        
        # Hitung Statistik Korelasi
        pearson_r,  _ = stats.pearsonr(df_bq2["days_diff"], df_bq2["score"])
        spearman_r, _ = stats.spearmanr(df_bq2["days_diff"], df_bq2["score"])

        # Hitung Rata-rata Skor per Grup (Late vs On Time)
        late_scores    = df_bq2[df_bq2["is_late"] == 1] ["score"]
        on_time_scores = df_bq2[df_bq2["is_late"] == 0] ["score"]
        
        # Hitung Proporsi Keterlambatan
        prop_late = df_bq2['is_late'].mean() * 100

        # --- METRIK UTAMA ---
        st.subheader("Statistik Utama")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pearson r", f"{pearson_r:.4f}", help="Korelasi linier antara keterlambatan dan skor.")
        col2.metric("Spearman r", f"{spearman_r:.4f}", help="Korelasi non-parametrik (ranking).")
        col3.metric("Proporsi Terlambat", f"{prop_late:.1f}%", help="Persentase siswa yang submit di atas deadline.")

        st.markdown("---")

        # --- VISUALISASI ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Score: On Time vs Late")
            fig, ax = plt.subplots(figsize=(5, 4))
            
            # Mapping label status
            status_map = {0: "On Time / Early", 1: "Late"}
            df_bq2["submit_status"] = df_bq2["is_late"].map(status_map)
            
            # Boxplot
            sns.boxplot(
                data=df_bq2,
                x="submit_status", y="score",
                palette={"On Time / Early": "#2ecc71", "Late": "#e74c3c"},
                order=["On Time / Early", "Late"],
                showfliers=False,
                ax=ax
            )
            
            ax.set_xlabel("Status Submit")
            ax.set_ylabel("Skor")
            ax.set_title("Perbandingan Skor Berdasarkan Ketepatan Waktu")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Scatter: days_diff vs Score")
            fig, ax = plt.subplots(figsize=(5, 4))
            
            # Scatter plot per grup risiko
            for group, grp_df in df_bq2.groupby("risk_group"):
                ax.scatter(
                    grp_df["days_diff"], grp_df["score"],
                    alpha=0.15, s=8,
                    color=palette[group], label=group, edgecolors='none'
                )
            
            # Garis Regresi Linear
            m, b, _, _, _ = stats.linregress(df_bq2["days_diff"], df_bq2["score"])
            min_x, max_x = int(df_bq2["days_diff"].min()), int(df_bq2["days_diff"].max())
            x_range = range(min_x, max_x + 1)
            y_range = [m * x + b for x in x_range]
            
            ax.plot(x_range, y_range, color="navy", linewidth=2, label=f"Regresi (r={pearson_r:.2f})")
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="Deadline (0 hari)")
            
            ax.set_xlabel("Selisih Hari (days_diff)")
            ax.set_ylabel("Skor")
            ax.set_title("Hubungan Keterlambatan vs Skor")
            ax.legend(fontsize=8, markerscale=3)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # --- INSIGHT ---
        diff_score = on_time_scores.mean() - late_scores.mean()
        st.info(
            f"**Insight :** Korelasi days_diff vs score secara statistik lemah "
            f"(Pearson r={pearson_r:.3f}), namun **pola perilaku** jelas terlihat.\n"
            f"- Siswa yang submit **tepat/lebih awal** rata-rata skor **{on_time_scores.mean():.1f}**.\n"
            f"- Siswa yang **terlambat** rata-rata skor **{late_scores.mean():.1f}**.\n"
            f"- **Selisih performa: {diff_score:.1f} poin.**\n"
            f"\n**Tindakan Rekomendasi:** Keterlambatan submit adalah *red flag* dini. Berikan intervensi segera jika `days_diff > 0`."
        )

        st.markdown("---")

    # =============================================================
    # LOGIKA BQ3 (Penurunan Aktivitas Demografi)
    # =============================================================
    elif bq_key == "bq3":
        st.header("Pertanyaan Bisnis 3: Penurunan Aktivitas per Kelompok Demografi")
        st.markdown(
            "_Kelompok demografi mana (`highest_education` atau `disability`) yang menunjukkan "
            "penurunan aktivitas interaksi VLE paling drastis selama 4 minggu pertama?_"
        )

        # Filter Demografi
        demo_option = st.selectbox(
            "Pilih Variabel Demografi untuk Analisis:",
            options=["highest_education", "disability"],
            key="bq3_demo_selector"
        )

        # Cek ketersediaan kolom
        if demo_option not in df_weekly.columns:
            st.warning(f"Kolom `{demo_option}` tidak ditemukan dalam data. Silakan pilih variabel lain atau periksa data.")
        else:
            # Persiapan Data
            # Group by student, module, presentation, demo_var, week
            df_student_bq3 = df_weekly.groupby(
                ["id_student", "code_module", "code_presentation", demo_option, "week"]
            )["total_clicks"].sum().reset_index()

            # Hitung rata-rata klik per minggu per grup demografi
            trend = df_student_bq3.groupby(
                [demo_option, "week"]
            )["total_clicks"].mean().reset_index()
            trend.columns = ["group", "week", "avg_clicks"]

            groups = trend["group"].unique()
            # Generate warna dinamis berdasarkan jumlah grup
            palette_bq3 = sns.color_palette("tab10", len(groups))

            # --- VISUALISASI ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Tren Klik per Minggu ({demo_option.replace('_', ' ').title()})")
                fig, ax = plt.subplots(figsize=(6, 4))
                for i, group in enumerate(groups):
                    grp_data = trend[trend["group"] == group]
                    # Hindari jika grup kosong
                    if not grp_data.empty:
                        ax.plot(
                            grp_data["week"], grp_data["avg_clicks"],
                            marker="o", label=str(group),
                            color=palette_bq3[i], linewidth=2
                        )
                
                # Garis penanda minggu 3 (biasanya lonjakan sebelum drop)
                ax.axvline(x=3, color="gray", linestyle="--", alpha=0.6, label="Batas Minggu 3 (Puncak)")
                ax.set_xlabel("Minggu")
                ax.set_ylabel("Rata-rata Total Clicks")
                ax.set_xticks([1, 2, 3, 4])
                ax.legend(fontsize=8, title=demo_option.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader(f"Risk Rate per {demo_option.replace('_', ' ').title()}")
                # Hitung rata-rata risk_label (1=Fail, 0=Pass) per grup demografi
                # Pastikan df_final punya kolom demo_option
                if demo_option in df_final.columns:
                    risk_rate_demo = df_final.groupby(demo_option)["risk_label"].mean().sort_values()
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    risk_rate_demo.plot(kind="barh", color="#e74c3c", ax=ax)
                    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.6, label="50% Threshold")
                    ax.set_xlabel("Proporsi Siswa Berisiko (Fail/Withdrawn)")
                    ax.set_ylabel(demo_option.replace('_', ' ').title())
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis="x")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning(f"Kolom `{demo_option}` tidak ada di df_final. Grafik risiko tidak ditampilkan.")

            # --- TABEL ANALISIS PENURUNAN ---
            st.subheader("Tabel Penurunan Aktivitas (Week 3 vs Week 4)")
            if not trend.empty:
                pivot = trend.pivot(index="group", columns="week", values="avg_clicks")
                
                # Hitung penurunan persentase
                # Drop dari minggu 3 ke 4
                if 3 in pivot.columns and 4 in pivot.columns:
                    pivot["Drop % (W3→W4)"] = ((pivot[3] - pivot[4]) / pivot[3] * 100).round(2)
                    # Drop dari minggu 1 ke 4
                    if 1 in pivot.columns:
                        pivot["Drop % (W1→W4)"] = ((pivot[1] - pivot[4]) / pivot[1] * 100).round(2)
                
                # Urutkan berdasarkan penurunan terbesar
                if "Drop % (W3→W4)" in pivot.columns:
                    pivot_sorted = pivot.sort_values(by="Drop % (W3→W4)", ascending=False)
                    
                    cols_to_show = [1, 2, 3, 4]
                    if "Drop % (W3→W4)" in pivot.columns: cols_to_show.append("Drop % (W3→W4)")
                    if "Drop % (W1→W4)" in pivot.columns: cols_to_show.append("Drop % (W1→W4)")
                    
                    st.dataframe(
                        pivot[cols_to_show].rename(columns={1: "W1", 2: "W2", 3: "W3", 4: "W4"}),
                        use_container_width=True
                    )
                    
                    # Cari ekstrem
                    worst_drop_group = pivot_sorted.index[0]
                    worst_drop_val = pivot_sorted["Drop % (W3→W4)"].iloc[0]
                    
                    # Cari kelompok risiko tertinggi
                    if demo_option in df_final.columns:
                        risk_demo = df_final.groupby(demo_option)["risk_label"].mean()
                        worst_risk_group = risk_demo.idxmax()
                        worst_risk_val = risk_demo.max()
                    else:
                        worst_risk_group, worst_risk_val = "N/A", 0

                    # --- INSIGHT ---
                    st.info(
                        f"**Insight Pertanyaan Bisnis 3:** Berdasarkan **{demo_option.replace('_', ' ')}**:\n"
                        f"1. **Penurunan Aktivitas Terbesar:** Kelompok **{worst_drop_group}** mengalami penurunan drastis "
                        f"sebesar **{worst_drop_val:.1f}%** dari minggu 3 ke 4.\n"
                        f"2. **Risiko Tertinggi:** Kelompok dengan risiko gagal tertinggi adalah **{worst_risk_group}** "
                        f"({worst_risk_val*100:.1f}%).\n"
                        f"**Tindakan Rekomendasi:** Fokuskan intervensi pada kelompok {worst_drop_group} di minggu ke-4 untuk mencegah drop-out."
                    )
                else:
                    st.warning("Data tidak cukup untuk menghitung tren penurunan mingguan.")
            else:
                st.warning("Tidak ada data tren yang tersedia.")

        st.markdown("---")

# =============================================================
# HALAMAN: MODEL MLM
# =============================================================
elif choice == "Model MLM":
    st.title("Hasil & Evaluasi Model Machine Learning")
    st.markdown("Analisis performa model prediktif untuk deteksi dini siswa berisiko.")
    st.markdown("---")

    # --- BAGIAN 1: RINGKASAN METRIK ---
    st.subheader("Ringkasan Performa Model (Gradient Boosting)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUC-ROC (Best)", "0.7212", help="Gradient Boosting Weighted")
    with col2:
        st.metric("Recall (Deteksi Risiko)", "62.5%", help="Persentase siswa berisiko yang berhasil dideteksi")
    with col3:
        st.metric("Precision", "62.6%", help="Akurasi prediksi risiko")
    with col4:
        st.metric("Accuracy", "66.0%", help="Total prediksi benar")

    st.markdown("---")

    # --- BAGIAN 2: GALLERY GAMBAR DALAM DROPDOWN ---
    st.subheader("Detail Visualisasi Model")
    st.markdown("Klik pada kategori di bawah untuk melihat gambar detail.")

    # --- PENCARIAN FOLDER GAMBAR---
    # Mencoba berbagai kemungkinan path berdasarkan struktur 'code\img\mlm'
    possible_folders = [
        "data/img/mlm", 
        "data/img/mlm", 
        "code/img", 
        "img", 
        "../code/img/mlm", 
        "../img/mlm"
    ]
    
    img_folder = None
    for folder in possible_folders:
        if os.path.exists(folder):
            # Cek apakah ada file png di folder ini
            files = os.listdir(folder)
            if any(f.endswith('.png') for f in files):
                img_folder = folder
                # st.success(f" Folder gambar ditemukan di: `{img_folder}`") #debug
                break
    
    if img_folder is None:
        st.error("❌ **Folder gambar tidak ditemukan!**")
        st.markdown("Pastikan folder `code/img/mlm` atau `img/mlm` ada di direktori proyek Anda.")
        st.info("Gambar yang dicari: `01_test_performance.png`, `02_treshold_tuning.png`, dll.")
        st.stop()

    # CSS untuk frame hitam
    css_frame = """
    <style>
    .mlm-img-frame {
        border: 3px solid black;
        background-color: white;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .mlm-img-frame img {
        max-width: 100%;
        height: auto;
        display: block;
    }
    </style>
    """
    st.markdown(css_frame, unsafe_allow_html=True)

    # --- FUNGSI BANTU RENDER GAMBAR ---
    def render_image_in_frame(img_name, folder):
        img_path = os.path.join(folder, img_name)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    base64_img = base64.b64encode(f.read()).decode()
                
                # Bersihkan judul: hapus .png, ganti underscore dengan spasi, Capitalize
                title = img_name.replace('.png', '').replace('_', ' ').title()
                
                st.markdown(f"#### {title}")
                st.markdown(f"""
                <div class="mlm-img-frame">
                    <img src="data:image/png;base64,{base64_img}">
                </div>
                """, unsafe_allow_html=True)
                return True
            except Exception as e:
                st.error(f"Gagal memuat gambar `{img_name}`: {e}")
                return False
        else:
            st.warning(f"File tidak ditemukan: `{img_path}`")
            return False

    # --- DROPDOWN 1: Performa & Threshold ---
    with st.expander("1. Performa Model & Threshold Tuning", expanded=False):
        st.markdown("### Evaluasi Komparatif & Penentuan Threshold")
        images_part1 = [
            "01_test_performance.png",
            "02_treshold_tuning.png"
        ]
        for img in images_part1:
            render_image_in_frame(img, img_folder)

    # --- DROPDOWN 2: Model & Evaluasi ---
    with st.expander("2. Arsitektur Model & ROC Curve", expanded=False):
        st.markdown("### Detail Gradient Boosting & Kurva ROC")
        images_part2 = [
            "03_gradient_boosting.png",
            "04_ROC Curve Performance.png"  # Perhatikan spasi di nama file
        ]
        for img in images_part2:
            render_image_in_frame(img, img_folder)

    # --- DROPDOWN 3: Feature Importance ---
    with st.expander("3. Fitur Paling Penting (Feature Importance)", expanded=False):
        st.markdown("### Analisis Kontribusi Fitur terhadap Prediksi")
        images_part3 = [
            "05_top_15_feature_importance.png"
        ]
        for img in images_part3:
            render_image_in_frame(img, img_folder)

    st.markdown("---")
    
    # --- BAGIAN 3: INSIGHT TEKS ---
    with st.expander("Baca Insight Strategis & Rekomendasi"):
        st.markdown("""
        **1. Performa Model:** Gradient Boosting (Weighted) unggul dengan AUC 0.7212.
        **2. Threshold Tuning:** 
           - Threshold 0.55 (Default) untuk efisiensi biaya.
           - Threshold 0.30 untuk keselamatan siswa (Recall 90%).
        **3. Fitur Penting:** 
           - `click_w4` (Aktivitas Minggu 4) adalah prediktor terkuat.
           - `is_late` (Keterlambatan Tugas) adalah sinyal bahaya dini.
           - Perilaku lebih penting daripada demografi.
        """)
