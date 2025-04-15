from flask import Flask, request, jsonify, send_from_directory
import torchaudio
import torch
from pyannote.audio import Pipeline
from whisper import load_model
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load diarization + transcription pipeline once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)
diarization_pipeline.to(device)
asr_model = load_model("base")

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'VrIndex.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.wav'
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)

    waveform, sr = torchaudio.load(file_path)
    diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sr})

    speaker_text = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = waveform[:, int(turn.start * sr): int(turn.end * sr)]
        segment_file = os.path.join(UPLOAD_FOLDER, f"seg_{speaker}_{turn.start:.2f}_{turn.end:.2f}.wav")
        torchaudio.save(segment_file, segment, sr)
        result = asr_model.transcribe(segment_file)
        text = result["text"]
        speaker_text.setdefault(speaker, "")
        speaker_text[speaker] += f" {text}"

    return jsonify({"speakers": speaker_text})

if __name__ == "__main__":
    app.run(debug=True)
