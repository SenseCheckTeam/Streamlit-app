import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import joblib
import streamlit as st
import json


@st.cache_resource
def load_model():
    model = XLMRobertaForSequenceClassification.from_pretrained("model")
    tokenizer = XLMRobertaTokenizer.from_pretrained("model")
    label_encoder = joblib.load("label_encoder.pkl")
    with open("label_map.json") as f:
        label_map = json.load(f)
    return model.eval(), tokenizer, label_encoder, label_map


model, tokenizer, le, label_map = load_model()

# Mapping diagnosis ke kategori
diagnosis_to_category = {
    "Katarak": "mata",
    "Konjungtivitis": "mata",
    "Mata Minus (Miopi)": "mata",
    "Otitis Eksterna": "telinga",
    "Rinitis Alergi": "hidung",
    "Sinusitis": "hidung",
    "Dermatitis Kontak": "kulit",
    "Psoriasis": "kulit",
    "Herpes Zoster (Cacar Shingles)": "kulit",
    "Flu": "hidung",  # Flu bisa dikaitkan dengan hidung
}

st.title("🔍 Diagnosa Awal Berdasarkan Gejala")
kategori = st.selectbox(
    "Pilih kategori pancaindra", ["mata", "telinga", "hidung", "kulit", "lidah"]
)
gejala = st.text_area("Tuliskan deskripsi gejala kamu", "")

if st.button("Prediksi Diagnosis"):
    if not gejala.strip():
        st.warning("Silakan tuliskan deskripsi gejala terlebih dahulu.")
    else:
        input_text = f"Kategori: {kategori} | Gejala: {gejala}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        if conf < 0.5:
            st.subheader("🩺 Diagnosis: KAMU HOAX")
            st.write(
                f"📌 Kemungkinan terlalu rendah (**{conf*100:.2f}%**) untuk diagnosis yang valid."
            )
        else:
            diagnosis_info = label_map.get(
                str(pred),
                {
                    "diagnosis": le.inverse_transform([pred])[0],
                    "saran": "Saran belum tersedia.",
                },
            )
            predicted_diagnosis = diagnosis_info["diagnosis"]
            predicted_category = diagnosis_to_category.get(
                predicted_diagnosis, "tidak diketahui"
            )

            if predicted_category != kategori:
                st.subheader(
                    "🚫 Maaf, diagnosis tidak sesuai dengan kategori yang kamu pilih."
                )
                st.write(
                    f"🔎 Diagnosis terdeteksi: **{predicted_diagnosis}** untuk kategori *{predicted_category}*."
                )
                st.write(f"📌 Kemungkinan: **{conf*100:.2f}%**")
            else:
                st.subheader(
                    f"🩺 Diagnosis untuk kategori *{kategori}*: {predicted_diagnosis}"
                )
                st.write(f"📌 Kemungkinan: **{conf*100:.2f}%**")
                st.write(f"💊 Saran Penanganan Pertama: {diagnosis_info['saran']}")
