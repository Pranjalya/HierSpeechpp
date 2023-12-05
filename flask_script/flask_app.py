from flask import Flask, request, jsonify, send_from_directory
from pydub import AudioSegment
import numpy as np
import io
import torch
import subprocess
from uuid import uuid4
from scipy.io import wavfile
from tempfile import NamedTemporaryFile
from inference_class import HierspeechSynthesizer


app = Flask(__name__)


@app.route('/get_audio', methods=['POST'])
def get_audio():
    try:
        # Get data from the request
        text = request.form.get('text')
        prompt = request.files['audio']
        ttv_temperature = float(request.form.get('ttv_temperature', 0.333))
        vc_temperature = float(request.form.get('vc_temperature', 0.333))
        duration_temperature = float(request.form.get('duration_temperature', 1.0))
        duration_length = float(request.form.get('duration_length', 1.0))
        denoise_ratio = float(request.form.get('denoise_ratio', 1))
        random_seed = int(request.form.get('denoise_ratio', 0))
        duration_only = True

        with NamedTemporaryFile(suffix=".wav") as f:
            sr, waveform = syn.synthesize(
                text=text, 
                prompt=prompt, 
                ttv_temperature=ttv_temperature,
                vc_temperature=vc_temperature,
                duration_temperature=duration_temperature,
                duration_length=duration_length,
                denoise_ratio=denoise_ratio,
                random_seed=random_seed,
                duration_only=duration_only, 
                vec2wav_prompt=None
            )

            wavfile.write(f, sr, waveform)
            mp3_file = f.name.replace('.wav', '.mp3')
            name = f"{uuid4()}.mp3"

            subprocess.run(f"ffmpeg -i {f.name} {mp3_file}", shell=True)

            # Return the processed audio as a response
            return send_from_directory(
                mp3_file,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name=name
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_duration', methods=['POST'])
def get_duration():
    try:
        # Get data from the request
        text = request.form.get('text')
        prompt = request.files['audio']
        ttv_temperature = float(request.form.get('ttv_temperature', 0.333))
        vc_temperature = float(request.form.get('vc_temperature', 0.333))
        duration_temperature = float(request.form.get('duration_temperature', 1.0))
        duration_length = float(request.form.get('duration_length', 1.0))
        denoise_ratio = float(request.form.get('denoise_ratio', 0))
        random_seed = int(request.form.get('denoise_ratio', 0))
        duration_only = False

        duration = syn.synthesize(
            text=text, 
            prompt=prompt, 
            ttv_temperature=ttv_temperature,
            vc_temperature=vc_temperature,
            duration_temperature=duration_temperature,
            duration_length=duration_length,
            denoise_ratio=denoise_ratio,
            random_seed=random_seed,
            duration_only=duration_only, 
            vec2wav_prompt=None
        )

        return jsonify({"duration": duration}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    syn = HierspeechSynthesizer(
        ckpt="./logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth",
        ckpt_text2w2v="./logs/ttv_libritts_v1/ttv_lt960_ckpt.pth",
        ckpt_sr="./speechsr24k/G_340000.pth",
        ckpt_sr48="./speechsr48k/G_100000.pth",
        denoiser_ckpt="denoiser/g_best",
        load_denoiser=True,
        device=device,
        seed=1111,
        output_dir="outputs"
    )
    app.run(debug=True)
