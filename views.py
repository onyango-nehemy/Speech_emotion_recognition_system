from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import os

# Load the model and label encoder when the server starts
model = load_model('model/simple_model.h5')  # Update this path to your model file
label_encoder = joblib.load('model/label_encoder.joblib')  # Update this path to your label encoder file

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        
        # Check if passwords match
        if password1 != password2:
            return HttpResponse("Passwords do not match!")  # Provide feedback for mismatched passwords
        
        try:
            user = User.objects.create_user(username=username, password=password1)
            user.save()  # Save the user to the database
            login(request, user)  # Log the user in
            return redirect('index')  # Redirect to the index page after registration
        except Exception as e:
            return HttpResponse(f"Error: {e}")  # Handle registration errors

    return render(request, 'emotion_recognition/register.html')


def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return HttpResponse("Invalid credentials")
    return render(request, 'emotion_recognition/login.html')


@login_required
def index(request):
    return render(request, 'emotion_recognition/index.html')


@login_required
def predict_emotion(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        file_path = os.path.join(fs.location, filename)

        # Load and preprocess the audio file
        features = extract_features(file_path)

        # Make a prediction
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions, axis=1)
        predicted_emotion = label_encoder.inverse_transform(predicted_label)

        result = predicted_emotion[0]  # Get the predicted emotion

        # Clean up the saved audio file
        os.remove(file_path)

        return JsonResponse({'result': result})  # Return as JSON response

    return JsonResponse({'result': 'Invalid request'}, status=400)


def extract_features(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)  # Load audio file with specified sampling rate
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    max_frames = 130
    if mel_spectrogram_db.shape[1] < max_frames:
        padding = max_frames - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, padding)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_frames]

    mel_spectrogram_db = mel_spectrogram_db[np.newaxis, ..., np.newaxis]  # Reshape for model input
    return mel_spectrogram_db