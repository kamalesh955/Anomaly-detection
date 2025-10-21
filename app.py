import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from torch import nn
import torchvision.models.video as video_models
import speech_recognition as sr
import pyttsx3
import random
from werkzeug.utils import secure_filename

import tempfile
import traceback
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
import tempfile
import traceback
import subprocess

# Define threshold consistent with training
threshold = 0.5

# Initialize speech recognizer and TTS engine
r = sr.Recognizer()
r.pause_threshold = 10
r.phrase_threshold = 0.3
r.non_speaking_duration = 0.8
engine = pyttsx3.init()

# Model class
class HierarchicalVideoClassifierR2plus1D(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        base = video_models.r2plus1d_18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features

        self.head_binary = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )
        self.head_anomaly = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 5)
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        out_bin = self.head_binary(feats)
        out_anom = self.head_anomaly(feats)
        return out_bin, out_anom

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalVideoClassifierR2plus1D(pretrained=False).to(device)
try:
    model.load_state_dict(torch.load('best_r2p1d_hierarchical.pth', map_location=device))
    print("Model loaded successfully")
except RuntimeError as e:
    print(f"Error loading state_dict: {e}")
model.eval()

classes = ["accident", "robbery", "fighting", "explosion", "shooting", "normal"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989]),
])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'powerhouse'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'wav', 'webm', 'ogg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ✅ FIX: make user available to all templates
@app.context_processor
def inject_user():
    return dict(user=current_user)

# In-memory user store
users = {'admin': {'password_hash': generate_password_hash('adminpass123'), 'role': 'admin'}}

class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.role = users[username]['role']

@login_manager.user_loader
def load_user(username):
    if username in users:
        return User(username)
    return None

# Login form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_frames(video_path, clip_len=16, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    max_start = max(0, total_frames - clip_len * frame_skip)
    start = random.randint(0, max_start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(clip_len):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
        for _ in range(frame_skip - 1):
            cap.read()
    cap.release()
    while len(frames) < clip_len:
        frames.append(frames[-1])
    frames = torch.stack(frames)
    frames = frames.permute(1, 0, 2, 3)
    return frames.unsqueeze(0)

def listen_once():
    try:
        with sr.Microphone() as source:
            print("🎤 Listening... Speak now!")
            r.adjust_for_ambient_noise(source, duration=1.0)
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
            text = r.recognize_google(audio).lower()
            print(f"📝 You said: {text}")
            engine.say(text)
            engine.runAndWait()
            return text
    except sr.RequestError as e:
        print(f"❌ Could not request results: {e}")
        return None
    except sr.UnknownValueError:
        print("❓ Could not understand audio")
        return None
    except sr.WaitTimeoutError:
        print("⏰ Timeout")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        if username in users and check_password_hash(users[username]['password_hash'], form.password.data):
            user = User(username)
            login_user(user)
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', form=form, error='Invalid username or password')
    return render_template('login.html', form=form)

# Logout route
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    print(f"Logout method: {request.method}")
    logout_user()
    return redirect(url_for('index'))

# Admin dashboard
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    all_data = {
        'video_predictions': [],
        'audio_transcriptions': []
    }
    return render_template('admin_dashboard.html', data=all_data)

# Video prediction
@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        frames = extract_frames(video_path)
        if frames is None:
            os.remove(video_path)
            return jsonify({'error': 'Failed to process video'})

        with torch.no_grad():
            frames = frames.to(device)
            out_bin, out_anom = model(frames)
            p_bin = torch.softmax(out_bin, dim=1)[0, 1].item()
            if p_bin >= threshold:
                p_anom = torch.softmax(out_anom, dim=1)[0]
                pred_idx = torch.argmax(p_anom).item()
                pred_class = classes[pred_idx]
                confidence = p_anom[pred_idx].item()
            else:
                pred_class = "normal"
                confidence = 1 - p_bin

        os.remove(video_path)
        return jsonify({'prediction': pred_class, 'confidence': f"{confidence:.2%}", 'input_type': 'video'})
    return jsonify({'error': 'Invalid file type'})

# Audio prediction
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    def _safe_remove(path):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    if "audio" not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    audio_file = request.files["audio"]
    if not audio_file or audio_file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(audio_file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".wav":
        return jsonify({"error": "Only WAV files accepted"}), 400

    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            audio_file.save(f.name)
            tmp_wav = f.name

        if os.path.getsize(tmp_wav) == 0:
            _safe_remove(tmp_wav)
            return jsonify({"error": "Empty file"}), 400

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            return jsonify({"text": text})
        except sr.UnknownValueError:
            return jsonify({"error": "Unintelligible speech"}), 200
        except sr.RequestError as e:
            return jsonify({"error": f"Recognition error: {e}"}), 500
    except Exception as e:
        tb = traceback.format_exc()
        print("=== /predict_audio ERROR ===")
        print(tb)
        return jsonify({"error": "Server error", "detail": str(e)}), 500
    finally:
        _safe_remove(tmp_wav)

@app.route('/result')
def result():
    pred = request.args.get('pred', 'Unknown')
    conf = request.args.get('conf', '0%')
    input_type = request.args.get('input_type', 'Unknown')
    return render_template('result.html', prediction=pred, confidence=conf, input_type=input_type)

@app.route('/audio_result')
def audio_result():
    text = request.args.get('text', 'No text recognized')
    return render_template('audio_result.html', text=text)

@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return "POST not handled on /", 405
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)