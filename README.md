# Pyannote.Web

A web application that integrates **pyannote.audio** for speaker diarization and transcription — enabling users to upload audio files and get segmented speaker labels and text output.

This project bridges **state-of-the-art speaker diarization** with a user-friendly interface, making it useful for meeting analysis, podcast segmentation, and automated speaker labeling.

---

## 🚀 Project Overview

Speaker diarization answers:
> “Who spoke when?”

By combining:
- Advanced ML models (pyannote)
- Audio processing
- Web frontend
- Backend API

This repo demonstrates a **full stack voice processing system** with:
✔ File upload UI  
✔ Back-end audio processing  
✔ Speaker segmentation  
✔ Transcript integration

---

## 🧠 Key Features

| Feature | Description |
|---------|-------------|
| Upload audio files | Users can upload .wav/.mp3 |
| Speaker diarization | Segment audio by speakers |
| Transcript generation | Optional conversion to text |
| JSON output | Structured timestamps & speaker labels |
| Web interface | Simple UI for interaction |

---

## 🛠️ Tech Stack

- **Python (Flask or FastAPI)** – Backend web server
- **pyannote.audio** – Speaker diarization
- **Whisper / other STT (optional)** – Transcription
- **HTML/CSS/JS** – Frontend
- **Web Upload & API endpoints**

---

## 📦 Installation

Clone the repository:
```
git clone https://github.com/Hari7383/Pyannote.web.git
cd Pyannote.web
```

Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
```
