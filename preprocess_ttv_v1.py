import os
import torch
import random
import torchaudio
import numpy as np
from tqdm import tqdm

import transformers
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

from utils import get_hparams_from_file
from ttv_v1.text import text_to_sequence
# from Mels_preprocess import MelSpectrogramFixed

from concurrent.futures import ThreadPoolExecutor


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7): 
        """we use the intermediate features of MMS.
           More specifically, we used the output from the 7th layer of the 24-layer transformer encoder.
        """
        super().__init__()
 
        self.mms = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m") 

        for param in self.mms.parameters():
            param.requires_grad = False
            param.grad = None
        self.mms.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x):
        outputs = self.mms(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]   
        y = y.permute((0, 2, 1))   
        return y


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.numpy().astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0, 'f0_max':1100})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]] 
    f0 = np.vstack(f0s)
    return f0


def preprocess(audio_file, text, hps, f0_path, w2v_path, text_path):
    # audio, sr = torchaudio.load(audio_file)
    if os.path.exists(text_path):
        text_for_ctc = text_to_sequence(text, hps.data.text_cleaners)
        speaker_dir, _ = os.path.split(text_path)
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir, exist_ok=True)
        torch.save(torch.tensor(text_for_ctc), text_path)

    if not os.path.exists(f0_path):
        speaker_dir, _ = os.path.split(f0_path)
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir, exist_ok=True)
        try:
            f0 = get_yaapt_f0(audio.numpy(), rate=16000)
        except:
            f0 = np.zeros((1, 1, audio.shape[-1] // 80))
        torch.save(torch.FloatTensor(f0.astype(np.float32).squeeze(0)), f0_path)

    if not os.path.exists(w2v_path):
        with torch.no_grad():
            speaker_dir, _ = os.path.split(w2v_path)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir, exist_ok=True)
            y_pad = torch.nn.functional.pad(audio, (40, 40), "reflect")
            wav2vecs = w2v(y_pad.cuda())
            torch.save(wav2vecs, w2v_path)


def process_data(d):
    audio_path, text, speaker = d.split("|")
    filepath = os.path.join(audio_16kHz_path, speaker, os.path.basename(audio_path))
    filepaths.append(filepath)
    f0_path = os.path.join(dump_path, "f0_yaapt", speaker, os.path.basename(audio_path).replace(".flac", ".f0.pt"))
    f0_paths.append(f0_path)
    w2v_path = os.path.join(dump_path, "mms_7", speaker, os.path.basename(audio_path).replace(".flac", ".w2v.pt"))
    w2v_paths.append(w2v_path)
    text_path = os.path.join(dump_path, "tokens", speaker, os.path.basename(audio_path).replace(".flac", ".txt.pt"))
    text_paths.append(text_path)
    
    # if not (os.path.exists(f0_path) and os.path.exists(w2v_path) and os.path.exists(text_path)):
    preprocess(filepath, text, hps, f0_path, w2v_path, text_path)



if __name__=="__main__":
    dump = "train"

    metadata_path = f"/root/dev/xtts_hindi_ft_dataset/metadata_{dump}.csv"
    dump_path = "/root/dev/xtts_hindi_ft_dataset/hierspeechpp_dump"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = "ttv_v1/config_hindi.json"

    audio_16kHz_path = "/root/dev/xtts_hindi_ft_dataset/audio_16kHz"

    os.makedirs(dump_path, exist_ok=True)
    os.makedirs(os.path.join(dump_path, "filelists"), exist_ok=True)
    os.makedirs(os.path.join(dump_path, "f0_yaapt"), exist_ok=True)
    os.makedirs(os.path.join(dump_path, "tokens"), exist_ok=True)
    os.makedirs(os.path.join(dump_path, "mms_7"), exist_ok=True)
    
    hps = get_hparams_from_file(config_file)

    w2v = Wav2vec2().to(device)

    with open(metadata_path) as f:
        data = f.read().splitlines()[1:]

    random.shuffle(data)

    filepaths = []
    f0_paths = []
    w2v_paths = []
    text_paths = []

    # for d in tqdm(data):
    #     audio_path, text, speaker = d.split("|")
    #     filepath = os.path.join(audio_16kHz_path, speaker, os.path.basename(audio_path))
    #     filepaths.append(filepath)
    #     f0_path = os.path.join(dump_path, "f0_yaapt", speaker, os.path.basename(audio_path).replace(".flac", ".f0.pt"))
    #     f0_paths.append(f0_path)
    #     w2v_path = os.path.join(dump_path, "mms_7", speaker, os.path.basename(audio_path).replace(".flac", ".w2v.pt"))
    #     w2v_paths.append(w2v_path)
    #     text_path = os.path.join(dump_path, "tokens", speaker, os.path.basename(audio_path).replace(".flac", ".txt.pt"))
    #     text_paths.append(text_path)
    #     if not (os.path.exists(f0_path) and os.path.exists(w2v_path) and os.path.exists(text_path)):
    #         preprocess(filepath, text, hps, f0_path, w2v_path, text_path)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        list(tqdm(executor.map(process_data, data), total=len(data)))

    with open(os.path.join(dump_path, "filelists", f"{dump}_wav.txt"), "w") as f:
        f.write("\n".join(filepaths))
    with open(os.path.join(dump_path, "filelists", f"{dump}_f0.txt"), "w") as f:
        f.write("\n".join(f0_paths))
    with open(os.path.join(dump_path, "filelists", f"{dump}_token.txt"), "w") as f:
        f.write("\n".join(text_paths))
    with open(os.path.join(dump_path, "filelists", f"{dump}_w2v.txt"), "w") as f:
        f.write("\n".join(w2v_paths))