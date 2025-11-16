# # image_caption_with_ecg.py
# import os
# from pathlib import Path
# from PIL import Image
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # ECG imports
# import wfdb
# import numpy as np
# import scipy.signal as sps
# import neurokit2 as nk


# import google.generativeai as genai
# genai.configure(api_key="")


# # ---------------- CONFIG ----------------
# MODEL_NAME = "Salesforce/blip-image-captioning-base"   # BLIP v1 base
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MAX_LENGTH = 40

# # Hardcoded files in SAME folder (change to your filenames)
# IMAGE_NAME = "data/ecg/mitbih/WhatsApp Image 2025-11-15 at 10.03.37 PM.jpeg"              # image file (jpg/png) in same folder
# ECG_RECORD_PREFIX = "100"             # wfdb record prefix (e.g., '100' -> 100.dat + 100.hea), located in a subfolder 'ecg' (see below)
# ECG_CSV = None                        # or set to "ecg1.csv" if you have a raw CSV (single-column samples)
# ECG_SUBFOLDER = "data/ecg/mitbih"                 # folder inside script dir where wfdb records or csv are kept
# # ----------------------------------------

# # ---------- BLIP (image caption) ----------
# def load_blip(model_name=MODEL_NAME, device=DEVICE):
#     print(f"Loading BLIP model {model_name} on {device} ...")
#     processor = BlipProcessor.from_pretrained(model_name)
#     model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
#     model.eval()
#     return processor, model

# def open_image(path):
#     img = Image.open(path)
#     img = img.convert("RGB")  # BLIP expects RGB
#     return img

# def caption_single_image(image_path, processor, model, device=DEVICE, max_length=MAX_LENGTH):
#     img = open_image(image_path)
#     inputs = processor(images=img, return_tensors="pt").to(device)
#     generated_ids = model.generate(
#         **inputs,
#         max_length=max_length,
#         num_beams=4,
#         early_stopping=True
#     )
#     caption = processor.decode(generated_ids[0], skip_special_tokens=True)
#     return caption

# # ---------- ECG -> text descriptor ----------
# def ecg_to_text_descriptor_from_wfdb(record_path_prefix):
#     """
#     record_path_prefix: full path prefix to WFDB record WITHOUT extension (e.g., "/path/to/100")
#     """
#     record = wfdb.rdrecord(record_path_prefix)
#     ecg = record.p_signal[:, 0]  # use first channel; adjust if needed
#     fs = record.fs

#     # bandpass filter
#     b, a = sps.butter(3, [0.5/(fs/2), 40/(fs/2)], btype='band')
#     ecg_filt = sps.filtfilt(b, a, ecg)

#     # process with neurokit2
#     signals, info = nk.ecg_process(ecg_filt, sampling_rate=fs)
#     rpeaks = info.get('ECG_R_Peaks', [])
#     if len(rpeaks) < 2:
#         return "ECG: insufficient beats detected to create descriptor."

#     rr_intervals = np.diff(rpeaks) / fs * 1000.0  # ms
#     avg_hr = 60000.0 / np.mean(rr_intervals)
#     rr_std = float(np.std(rr_intervals))

#     # rule-based label
#     label = "normal rhythm"
#     if avg_hr > 100:
#         label = "tachycardia"
#     elif avg_hr < 60:
#         label = "bradycardia"
#     if rr_std > 100:
#         # allow combined label
#         if "normal" in label:
#             label = "irregular rhythm (possible AFib)"
#         else:
#             label = f"{label}, irregular rhythm (possible AFib)"

#     descriptor = f"ECG shows {label}; avg heart rate {avg_hr:.0f} bpm; RR-interval std {rr_std:.1f} ms."
#     return descriptor

# def ecg_to_text_descriptor_from_array(ecg_array, fs=250):
#     """
#     ecg_array: 1D numpy array of samples
#     fs: sampling frequency (Hz)
#     """
#     ecg = np.asarray(ecg_array).flatten()
#     if ecg.size < fs:  # less than 1s
#         return "ECG: signal too short."

#     # bandpass
#     b, a = sps.butter(3, [0.5/(fs/2), 40/(fs/2)], btype='band')
#     ecg_filt = sps.filtfilt(b, a, ecg)

#     signals, info = nk.ecg_process(ecg_filt, sampling_rate=fs)
#     rpeaks = info.get('ECG_R_Peaks', [])
#     if len(rpeaks) < 2:
#         return "ECG: insufficient beats detected to create descriptor."

#     rr_intervals = np.diff(rpeaks) / fs * 1000.0  # ms
#     avg_hr = 60000.0 / np.mean(rr_intervals)
#     rr_std = float(np.std(rr_intervals))

#     label = "normal rhythm"
#     if avg_hr > 100:
#         label = "tachycardia"
#     elif avg_hr < 60:
#         label = "bradycardia"
#     if rr_std > 100:
#         if "normal" in label:
#             label = "irregular rhythm (possible AFib)"
#         else:
#             label = f"{label}, irregular rhythm (possible AFib)"

#     descriptor = f"ECG shows {label}; avg heart rate {avg_hr:.0f} bpm; RR-interval std {rr_std:.1f} ms."
#     return descriptor

# # ---------- Utility: combine into prompt ----------
# def generate_prompt(ecg_desc, image_caption, clinical_note=None):
#     parts = [
#         f"ECG findings: {ecg_desc}",
#         f"X-ray findings: {image_caption}"
#     ]
#     if clinical_note:
#         parts.append(f"Clinical note: {clinical_note}")
#     #parts.append("Task: Write a concise 2-3 sentence diagnostic-style summary integrating the ECG and X-ray findings.")
#     return " ".join(parts)

# def generate_clinical_interpretation_with_gemini(ecg_desc, image_caption):
#     genai.configure(api_key="AIzaSyCHdUg7jXUMNxeWV-IAgm0yObajtvMe-PA")  # Or load from env

#     # Build a very detailed prompt for Gemini
#     prompt = f"""
# You are a senior medical AI system. Perform an advanced multi-modal clinical interpretation.

# ---
# ### IMAGE CAPTION (Radiology Findings):
# {image_caption}

# ### ECG FINDINGS:
# {ecg_desc}

# ---

# ### Task:
# Generate a detailed 50–60 line medical-style interpretation including:

# 1. **ECG Analysis** (rate, rhythm, morphology, abnormalities)
# 2. **Radiology Interpretation** (structures, abnormalities, differential diagnosis)
# 3. **Integration of ECG + X-ray** (pathophysiology link)
# 4. **Possible Conditions** (explain reasoning)
# 5. **Clinical Correlation** (symptoms → findings)
# 6. **Complication Risks**
# 7. **Red-flag Warnings**
# 8. **Recommended Investigations**
# 9. **Initial Management Plan**
# 10. **Patient Counseling Advice**

# Write it in highly structured, clinical, professional format.
# Use bullet points, subsections, and medically accurate language.
# """

#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     return response.text

# # ---------- Main ----------
# def main():
#     script_dir = Path(__file__).resolve().parent

#     # Image path (hardcoded in same folder)
#     image_path = script_dir / IMAGE_NAME
#     if not image_path.exists():
#         print(f"❌ ERROR: Hardcoded image not found: {image_path}")
#         print("Make sure the image is in the same folder as this script.")
#         return
#     print(f"Using hardcoded image: {image_path}")

#     # ECG locations: first prefer WFDB files in ecg subfolder (prefix), else CSV if provided
#     ecg_folder = script_dir / ECG_SUBFOLDER
#     wfdb_prefix_path = ecg_folder / ECG_RECORD_PREFIX  # wfdb expects prefix without extension
#     csv_path = ecg_folder / ECG_CSV if ECG_CSV else None

#     # Load BLIP
#     processor, blip_model = load_blip()

#     # Caption image
#     print("Generating image caption with BLIP...")
#     try:
#         image_caption = caption_single_image(str(image_path), processor, blip_model)
#     except Exception as e:
#         print("Error during image captioning:", e)
#         image_caption = "[ERROR generating caption]"

#     print("\n✅ Image caption:")
#     print(image_caption)

#     # Create ECG descriptor
#     ecg_descriptor = None
#     if wfdb_prefix_path.exists() or (ecg_folder.exists() and any(wfdb_prefix_path.with_suffix(ext).exists() for ext in ['.dat','.hea','.raw','.atr'])):
#         # if prefix exists (we assume files like 100.dat and 100.hea are present)
#         try:
#             print("\nProcessing WFDB ECG record:", wfdb_prefix_path)
#             ecg_descriptor = ecg_to_text_descriptor_from_wfdb(str(wfdb_prefix_path))
#         except Exception as e:
#             print("Error reading WFDB record:", e)
#             ecg_descriptor = "ECG: error processing WFDB record."
#     elif csv_path and csv_path.exists():
#         try:
#             print("\nProcessing ECG from CSV:", csv_path)
#             arr = np.loadtxt(str(csv_path), delimiter=",")
#             # default fs: 250 Hz (change if your CSV uses different sampling)
#             ecg_descriptor = ecg_to_text_descriptor_from_array(arr, fs=250)
#         except Exception as e:
#             print("Error reading ECG CSV:", e)
#             ecg_descriptor = "ECG: error processing CSV."
#     else:
#         print("\nNo WFDB record or ECG CSV found in", ecg_folder)
#         print("Place a WFDB record (prefix files) named", ECG_RECORD_PREFIX, "or set ECG_CSV to a CSV filename.")
#         ecg_descriptor = "ECG: not provided."

#     print("\n✅ ECG descriptor:")
#     print(ecg_descriptor)

#     # Compose prompt (example)
#     prompt = generate_prompt(ecg_descriptor, image_caption, clinical_note=None)
#     print("\n----- Combined Prompt (for generator) -----")
#     print(prompt)
#     print("------------------------------------------")

#     print("\nCalling Gemini for final multimodal interpretation...")
#     try:
#         final_report = generate_clinical_interpretation_with_gemini(ecg_descriptor, image_caption)
#         print("\n\n===== FINAL MULTI-MODAL CLINICAL REPORT =====\n")
#         print(final_report)
#         print("\n=============================================\n")
#     except Exception as e:
#         print("❌ Gemini API Error:", e)

# if __name__ == "__main__":
#     main()



# trifusion_backend.py

import os
from PIL import Image
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import wfdb
import scipy.signal as sps
import neurokit2 as nk
import google.generativeai as genai


# ----------------------------- #
#       LOAD BLIP MODEL
# ----------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)
blip_model.eval()


def caption_image(image: Image.Image) -> str:
    """Generate caption using BLIP."""
    inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
    ids = blip_model.generate(**inputs, max_length=40, num_beams=4)
    return blip_processor.decode(ids[0], skip_special_tokens=True)


# ----------------------------- #
#       ECG PROCESSING
# ----------------------------- #
def ecg_to_text_descriptor_from_wfdb(prefix_path: str) -> str:
    """Takes wfdb record prefix and produces ECG descriptor."""
    record = wfdb.rdrecord(prefix_path)
    ecg = record.p_signal[:, 0]
    fs = record.fs

    # Filter
    b, a = sps.butter(3, [0.5/(fs/2), 40/(fs/2)], btype="band")
    ecg_filt = sps.filtfilt(b, a, ecg)

    # NeuroKit
    signals, info = nk.ecg_process(ecg_filt, sampling_rate=fs)
    rpeaks = info.get("ECG_R_Peaks", [])

    if len(rpeaks) < 2:
        return "ECG: insufficient peaks detected."

    # RR stats
    rr = np.diff(rpeaks) / fs * 1000.0
    avg_hr = 60000 / np.mean(rr)
    rr_std = float(np.std(rr))

    # Simple logic
    label = "normal rhythm"
    if avg_hr > 100:
        label = "tachycardia"
    elif avg_hr < 60:
        label = "bradycardia"

    if rr_std > 100:
        label += ", irregular rhythm (possible AFib)"

    return (
        f"ECG shows {label}; "
        f"avg heart rate {avg_hr:.0f} bpm; "
        f"RR-interval std {rr_std:.1f} ms."
    )


# ----------------------------- #
#       GEMINI MULTIMODAL REPORT
# ----------------------------- #
def generate_clinical_report(ecg_desc: str, image_caption: str, api_key: str) -> str:
    genai.configure(api_key=api_key)

    prompt = f"""
You are a senior medical AI system. Perform an advanced multi-modal clinical interpretation.

### Radiology (X-ray) Caption:
{image_caption}

### ECG Interpretation:
{ecg_desc}

---

### Task:
Generate a detailed, professionally structured **50–60 line multimodal clinical interpretation**, including:

1. **Detailed ECG analysis**
2. **Radiological analysis**
3. **Integrated diagnosis reasoning**
4. **Potential conditions**
5. **Clinical correlations**
6. **Complication risks**
7. **Red-flag warnings**
8. **Recommended investigations**
9. **Management plan**
10. **Patient counseling**

Use medical terminology and structured formatting.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    out = model.generate_content(prompt)
    return out.text
