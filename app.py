import os
import torch
import gdown
import cv2
import numpy as np
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from torch import nn
import torchvision.models.video as video_models
import speech_recognition as sr
from gtts import gTTS
import io
import base64
import random
import sqlite3
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template, redirect, url_for, g
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
threshold = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# SPEECH: Text-to-Speech Helper
# -----------------------------
def speak(text):
    """Convert text to speech using gTTS and return base64 audio string."""
    try:
        tts = gTTS(text)
        with io.BytesIO() as audio_fp:
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_bytes = audio_fp.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            return f"data:audio/mp3;base64,{audio_base64}"
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None

# -----------------------------
# MODEL SETUP AND CACHING
# -----------------------------
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

# Cache model in home directory
cache_dir = os.path.expanduser("~/.cache/emergency_model")
os.makedirs(cache_dir, exist_ok=True)

model_path = os.path.join(cache_dir, "best_r2p1d_hierarchical.pth")
drive_file_id = "1nN84_zyLeL7SEKd3u8JiXhPPvjMzCLUt"
drive_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    print("⬇️ Downloading model from Google Drive...")
    try:
        gdown.download(drive_url, model_path, quiet=False)
        print(f"✅ Model downloaded to {model_path}")
    except Exception as e:
        print(f"⚠️ Error downloading model: {e}")
else:
    print(f"✅ Using cached model at {model_path}")

model = HierarchicalVideoClassifierR2plus1D(pretrained=False).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

model.eval()

classes = ["accident", "robbery", "fighting", "explosion", "shooting", "normal"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989]),
])

# -----------------------------
# FLASK APP INITIALIZATION
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'powerhouse'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'wav', 'webm', 'ogg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# LOGIN SYSTEM
# -----------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

users = {'admin': {'password_hash': generate_password_hash('adminpass123'), 'role': 'admin'}}

class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.role = users[username]['role']

@login_manager.user_loader
def load_user(username):
    return User(username) if username in users else None

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@app.context_processor
def inject_user():
    return dict(user=current_user)

# -----------------------------
# DATABASE SETUP
# -----------------------------
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('emergency_response.db', check_same_thread=False)
        g.db.row_factory = sqlite3.Row
        cursor = g.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class TEXT,
                confidence REAL,
                input_type TEXT,
                latitude REAL,
                longitude REAL,
                timestamp DATETIME
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                input_type TEXT,
                latitude REAL,
                longitude REAL,
                timestamp DATETIME
            )
        ''')
        g.db.commit()
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# -----------------------------
# VIDEO PROCESSING
# -----------------------------
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
    frames = torch.stack(frames).permute(1, 0, 2, 3)
    return frames.unsqueeze(0)

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        if username in users and check_password_hash(users[username]['password_hash'], form.password.data):
            login_user(User(username))
            return redirect(url_for('admin_dashboard'))
        return render_template('login.html', form=form, error='Invalid username or password')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT class, confidence, input_type, latitude, longitude FROM predictions WHERE input_type = ?', ('video',))
    video_predictions = [{'class': row[0], 'confidence': f"{row[1]:.2%}", 'latitude': row[3], 'longitude': row[4]} for row in cursor.fetchall()]
    cursor.execute('SELECT text, latitude, longitude FROM transcriptions WHERE input_type = ?', ('audio',))
    audio_transcriptions = [{'text': row[0], 'latitude': row[1], 'longitude': row[2]} for row in cursor.fetchall()]
    return render_template('admin_dashboard.html', data={'video_predictions': video_predictions, 'audio_transcriptions': audio_transcriptions})

@app.route('/predict_video', methods=['POST'])
def predict_video():
    db = get_db()
    cursor = db.cursor()
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            latitude = request.form.get('latitude', type=float)
            longitude = request.form.get('longitude', type=float)

            frames = extract_frames(video_path)
            if frames is None:
                os.remove(video_path)
                return jsonify({'error': 'Failed to process video'}), 400

            with torch.no_grad():
                frames = frames.to(device)
                out_bin, out_anom = model(frames)
                p_bin = torch.softmax(out_bin, dim=1)[0, 1].item()
                if p_bin >= threshold:
                    p_anom = torch.softmax(out_anom, dim=1)[0]
                    pred_idx = torch.argmax(p_anom).item()
                    pred_class = classes[pred_idx]
                    confidence = p_anom[pred_idx].item()
                    alert_audio = speak(f"Alert! {pred_class} detected with {confidence*100:.1f} percent confidence.")
                else:
                    pred_class = "normal"
                    confidence = 1 - p_bin
                    alert_audio = speak("No anomaly detected. Situation normal.")

            cursor.execute('INSERT INTO predictions (class, confidence, input_type, latitude, longitude, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                           (pred_class, confidence, 'video', latitude, longitude, datetime.utcnow()))
            db.commit()
            os.remove(video_path)

            return jsonify({'prediction': pred_class, 'confidence': f"{confidence:.2%}", 'audio': alert_audio})
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error in predict_video: {e}")
        return jsonify({'error': 'Server error', 'detail': str(e)}), 500

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    db = get_db()
    cursor = db.cursor()
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
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            cursor.execute('INSERT INTO transcriptions (text, input_type, latitude, longitude, timestamp) VALUES (?, ?, ?, ?, ?)',
                           (text, 'audio', None, None, datetime.utcnow()))
            db.commit()
            alert_audio = speak(f"Received audio: {text}")
            return jsonify({"text": text, "audio": alert_audio})
        except sr.UnknownValueError:
            return jsonify({"error": "Unintelligible speech"}), 200
        except sr.RequestError as e:
            return jsonify({"error": f"Recognition error: {e}"}), 500
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": "Server error", "detail": str(e)}), 500
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
