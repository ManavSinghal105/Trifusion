# app.py       // this is the frontend code for the Trifusion app
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import wfdb

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

from final_code import (
    caption_image,
    ecg_to_text_descriptor_from_wfdb,
    generate_clinical_report
)


st.set_page_config(
    page_title="TriFusion: Multimodal Medical AI",
    layout="wide",
)


st.title("🩺 TriFusion – ECG + X-Ray Multimodal AI Assistant")

def generate_pdf(report_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    width, height = letter
    y = height - 50  # Start 50px from the top

    c.setFont("Helvetica", 11)

    for line in report_text.split("\n"):
        if y < 50:  # New page when space runs out
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 50

        c.drawString(40, y, line)
        y -= 18  # Move down line-by-line

    c.save()
    buffer.seek(0)
    return buffer


############################
# Gemini API Key Input
############################
api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password")
if not api_key:
    st.warning("Enter your Gemini API Key to continue.")
    st.stop()


############################
# File Uploaders
############################
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload Medical Image")
    xray_file = st.file_uploader("Upload X-ray (.jpg/.png)", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📈 Upload ECG Record (.dat + .hea)")
    ecg_dat = st.file_uploader("Upload .dat file", type=["dat"])
    ecg_hea = st.file_uploader("Upload .hea file", type=["hea"])


if st.button("🚀 Generate Clinical Report"):
    if not (xray_file and ecg_dat and ecg_hea):
        st.error("Upload BOTH X-ray and ECG (.dat + .hea).")
        st.stop()

    # ---------------- IMAGE PROCESSING ----------------
    img = Image.open(xray_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", width=330)

    with st.spinner("Generating BLIP caption..."):
        caption = caption_image(img)

    st.success("Image caption generated:")
    st.write(caption)

    # ---------------- ECG PROCESSING ------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        dat_path = os.path.join(tmpdir, "ecg.dat")
        hea_path = os.path.join(tmpdir, "ecg.hea")

        with open(dat_path, "wb") as f:
            f.write(ecg_dat.read())

        with open(hea_path, "wb") as f:
            f.write(ecg_hea.read())

        prefix_path = os.path.join(tmpdir, "100")

        # Rename written files to match WFDB prefix
        os.rename(dat_path, prefix_path + ".dat")
        os.rename(hea_path, prefix_path + ".hea")

        with st.spinner("Processing ECG..."):
            ecg_desc = ecg_to_text_descriptor_from_wfdb(prefix_path)

    st.success("ECG descriptor generated:")
    st.write(ecg_desc)

    # ---------------- GEMINI --------------------------
    with st.spinner("Generating clinical multi-modal report..."):
        report = generate_clinical_report(ecg_desc, caption, api_key)

    st.markdown("## 📝 Final Clinical Report")
    st.write(report)
    pdf_buffer = generate_pdf(report)

    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_buffer,
        file_name="clinical_report.pdf",
        mime="application/pdf"
)
