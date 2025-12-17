import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gradio as gr
import numpy as np
import tensorflow as tf
import librosa
import pretty_midi
import tempfile
import shutil

# ====== Paramètres ======
SR = 22000
HOP_LENGTH = 220
N_FFT = 2048
N_MELS = 128

# ====== Charger le modèle une seule fois ======
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("./model.keras", compile=False)
        print(f"✅ Model loaded: {model.name}")
    return model

def audio_to_mel_3d(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_db = (S_db - S_min) / (S_max - S_min + 1e-8)
    return S_db[..., np.newaxis]

def probs_to_onset_binary(onset_pred, threshold=0.85, min_distance=10):
    T, P = onset_pred.shape
    onset_binary = np.zeros_like(onset_pred, dtype=np.float32)
    for p in range(P):
        probs = onset_pred[:, p]
        t = 1
        last_t = -min_distance
        while t < T - 1:
            if (probs[t] >= threshold and probs[t] >= probs[t - 1] and
                probs[t] >= probs[t + 1] and t - last_t >= min_distance):
                onset_binary[t, p] = 1.0
                last_t = t
                t += min_distance
            else:
                t += 1
    return onset_binary

def onset_binary_to_midi(onset_binary, output_path, fps=100, pitch_min=21, min_duration=0.35, velocity=45):
    T, n_pitches = onset_binary.shape
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for p in range(n_pitches):
        t = 0
        while t < T:
            if onset_binary[t, p] >= 0.5:
                start_frame = t
                while t < T and onset_binary[t, p] >= 0.5:
                    t += 1
                end_frame = t
                start_time = start_frame / fps
                end_time = end_frame / fps
                if end_time - start_time < min_duration:
                    end_time = start_time + min_duration
                note = pretty_midi.Note(velocity=velocity, pitch=pitch_min + p, start=start_time, end=end_time)
                piano.notes.append(note)
            else:
                t += 1
    pm.instruments.append(piano)
    pm.write(output_path)

def generate_video_simple(midi_path, video_path, fps=30):
    """Génère une vidéo piano roll avec synthviz standard."""
    from synthviz import create_video
    
    create_video(
        input_midi=midi_path,
        video_filename=video_path,
        image_width=1280,
        image_height=720,
        fps=fps,
    )
    
    # Cleanup
    if os.path.exists("video_frames"):
        shutil.rmtree("video_frames")
    if os.path.exists("output.wav"):
        os.remove("output.wav")

def transcribe(audio_path, progress=gr.Progress()):
    """Pipeline complet de transcription."""
    if audio_path is None:
        return None, None, "❌ Please upload an audio file"
    
    try:
        progress(0.1, desc="Loading model...")
        model = load_model()
        
        progress(0.2, desc="Processing audio...")
        mel_3d = audio_to_mel_3d(audio_path)
        
        progress(0.4, desc="Running inference...")
        mel = np.squeeze(mel_3d, axis=-1).T
        mel_example = mel[np.newaxis, ...]
        onset_pred = model.predict(mel_example, verbose=0)[0]
        
        # Debug - vérifie si le problème Streamlit existe avec Gradio
        debug_msg = f"✅ onset_pred mean = {onset_pred.mean():.4f}"
        print(debug_msg)
        
        progress(0.6, desc="Generating MIDI...")
        onset_binary = probs_to_onset_binary(onset_pred)
        
        # Créer fichiers temporaires
        tmp_dir = tempfile.mkdtemp()
        midi_path = os.path.join(tmp_dir, "transcription.mid")
        video_path = os.path.join(tmp_dir, "piano_roll.mp4")
        
        onset_binary_to_midi(onset_binary, midi_path)
        
        progress(0.8, desc="Creating video...")
        generate_video_simple(midi_path, video_path)
        
        progress(1.0, desc="Done!")
        
        return video_path, midi_path, debug_msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# ====== Interface Gradio ======
with gr.Blocks(title="🎹 Piano AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎹 Piano AI
    Upload a piano audio file and get the MIDI transcription with a piano roll video.
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Piano Audio")
            btn = gr.Button("🎵 Transcribe", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        video_output = gr.Video(label="Piano Roll Video")
        midi_output = gr.File(label="Download MIDI")
    
    btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[video_output, midi_output, status]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
