# from flask import Flask, request, jsonify, send_from_directory
# import torchaudio
# import torch
# from pyannote.audio import Pipeline
# from whisper import load_model
# import os
# from datetime import datetime
# import ffmpeg


# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load speaker diarization pipeline
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diarization_pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="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"
# )
# diarization_pipeline.to(device)

# # Load Whisper transcription model
# asr_model = load_model("base")  # or use "small", "medium", "large" depending on your GPU

# @app.route('/')
# def index():
#     return send_from_directory('.', 'VrIndex.html')

# @app.route('/process-audio', methods=['POST'])
# def process_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file uploaded"}), 400

#     # audio_file = request.files['audio']
#     # filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.webm'
#     # file_path = os.path.join(UPLOAD_FOLDER, filename)
#     # audio_file.save(file_path)

#     # #return send_file(file_path, mimetype="audio/wav")

#     # # Load audio
#     # waveform, sr = torchaudio.load(file_path)

#     audio_file = request.files['audio']
#     filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.webm'
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     audio_file.save(file_path)

#     # Convert .webm to .wav
#     wav_path = file_path.replace(".webm", ".wav")
#     ffmpeg.input(file_path).output(wav_path).run(overwrite_output=True)

#     # Load the converted .wav file
#     waveform, sr = torchaudio.load(wav_path)


#     # Run speaker diarization
#     diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sr})

#     # Transcribe segments speaker-wise
#     speaker_text = {}
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         segment = waveform[:, int(turn.start * sr): int(turn.end * sr)]
#         segment_file = os.path.join(UPLOAD_FOLDER, f"seg_{speaker}_{int(turn.start*1000)}.wav")
#         torchaudio.save(segment_file, segment, sr)

#         # Transcribe using Whisper
#         result = asr_model.transcribe(segment_file)
#         text = result["text"]

#         if speaker not in speaker_text:
#             speaker_text[speaker] = ""
#         speaker_text[speaker] += f" {text}"
#     print("Received audio:", file_path)
#     print("File size (bytes):", os.path.getsize(file_path))


#     return jsonify({"speakers": speaker_text})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, send_from_directory
# import torchaudio
# import torch
# from pyannote.audio import Pipeline
# from whisper import load_model
# import os
# from datetime import datetime
# import ffmpeg  # ffmpeg-python

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load PyAnnote speaker diarization model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# diarization_pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="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"
# )
# diarization_pipeline.to(device)

# # Load Whisper model (base, or choose small/medium/large)
# asr_model = load_model("base")

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index.html')

# @app.route('/process-audio', methods=['POST'])
# def process_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file uploaded"}), 400

#     audio_file = request.files['audio']
#     webm_filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.webm'
#     webm_path = os.path.join(UPLOAD_FOLDER, webm_filename)
#     audio_file.save(webm_path)

#     # Convert .webm to .wav using ffmpeg
#     wav_path = webm_path.replace(".webm", ".wav")
#     try:
#         ffmpeg.input(webm_path).output(wav_path).run(overwrite_output=True)
#     except ffmpeg.Error as e:
#         return jsonify({"error": "FFmpeg conversion failed", "details": str(e)}), 500

#     # Load .wav audio
#     try:
#         waveform, sr = torchaudio.load(wav_path)
#     except Exception as e:
#         return jsonify({"error": "Could not load WAV file", "details": str(e)}), 500

#     # Run diarization
#     diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sr})

#     # Transcribe segments
#     speaker_text = {}
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         segment = waveform[:, int(turn.start * sr): int(turn.end * sr)]
#         segment_file = os.path.join(UPLOAD_FOLDER, f"seg_{speaker}_{int(turn.start*1000)}.wav")
#         torchaudio.save(segment_file, segment, sr)

#         result = asr_model.transcribe(segment_file)
#         text = result["text"]

#         if speaker not in speaker_text:
#             speaker_text[speaker] = ""
#         speaker_text[speaker] += f" {text}"

#     return jsonify({"speakers": speaker_text})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, send_from_directory #type: ignore
import torchaudio #type: ignore
import torch #type: ignore
from pyannote.audio import Pipeline #type: ignore
from whisper import load_model #type: ignore
import os
from datetime import datetime
import ffmpeg #type: ignore

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load diarization pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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"
)
diarization_pipeline.to(device)

# Load Whisper ASR model
asr_model = load_model("base")  # Can change to "small", "medium", or "large"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    filename = datetime.now().strftime('%Y%m%d_%H%M%S')
    extension = os.path.splitext(audio_file.filename)[1].lower()
    input_path = os.path.join(UPLOAD_FOLDER, f"{filename}{extension}")
    audio_file.save(input_path)

    # Convert to .wav if file is .webm
    if extension == ".webm":
        wav_path = os.path.join(UPLOAD_FOLDER, f"{filename}.wav")
        try:
            ffmpeg.input(input_path).output(wav_path).run(overwrite_output=True)
        except ffmpeg.Error as e:
            return jsonify({"error": "FFmpeg conversion failed", "details": str(e)}), 500
    elif extension == ".wav":
        wav_path = input_path
    else:
        return jsonify({"error": f"Unsupported file type: {extension}"}), 400

    # Load audio
    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as e:
        return jsonify({"error": "Failed to load WAV file", "details": str(e)}), 500

    # Run speaker diarization
    try:
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sr})
    except Exception as e:
        return jsonify({"error": "Diarization failed", "details": str(e)}), 500

    # Transcribe segments speaker-wise
    speaker_text = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = waveform[:, int(turn.start * sr): int(turn.end * sr)]
        segment_file = os.path.join(UPLOAD_FOLDER, f"seg_{speaker}_{int(turn.start*1000)}.wav")
        torchaudio.save(segment_file, segment, sr)

        try:
            result = asr_model.transcribe(segment_file)
            text = result["text"]
        except Exception as e:
            text = "[Transcription failed]"

        speaker_text.setdefault(speaker, "")
        speaker_text[speaker] += f" {text}"

    return jsonify({"speakers": speaker_text})

if __name__ == '__main__':
    app.run(debug=True)
