import os
import torch
import numpy as np
from uuid import uuid4
from scipy.io.wavfile import write
import torchaudio
import utils
from Mels_preprocess import MelSpectrogramFixed

from hierspeechpp_speechsynthesizer import SynthesizerTrn
from ttv_v1.text import text_to_sequence
from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from speechsr48k.speechsr import SynthesizerTrn as SpeechSR48
from denoiser.generator import MPNet
from denoiser.infer import denoise


class HierspeechSynthesizer:
    def __init__(
        self,
        ckpt,
        ckpt_text2w2v,
        ckpt_sr=None,
        ckpt_sr48=None,
        denoiser_ckpt=None,
        scale_norm="max",
        output_sr=48000,
        load_denoiser=True,
        device="cpu",
        seed=1111,
        output_dir="output",
    ):
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, defaulting to CPU")
            self.device = torch.device("cpu")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        hps = utils.get_hparams_from_file(
            os.path.join(os.path.split(ckpt)[0], "config.json")
        )
        hps_t2w2v = utils.get_hparams_from_file(
            os.path.join(os.path.split(ckpt_text2w2v)[0], "config.json")
        )
        h_sr = utils.get_hparams_from_file(
            os.path.join(os.path.split(ckpt_sr)[0], "config.json")
        )
        h_sr48 = utils.get_hparams_from_file(
            os.path.join(os.path.split(ckpt_sr48)[0], "config.json")
        )
        hps_denoiser = utils.get_hparams_from_file(
            os.path.join(os.path.split(denoiser_ckpt)[0], "config.json")
        )

        self.mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate,
            n_fft=hps.data.filter_length,
            win_length=hps.data.win_length,
            hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin,
            f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels,
            window_fn=torch.hann_window,
        ).to(self.device)

        self.net_g = None
        self.text2w2v = None
        self.speechsr = None
        self.denoiser = None

        self.load_denoiser = load_denoiser

        self.load_models(
            ckpt,
            ckpt_text2w2v,
            ckpt_sr48=ckpt_sr48,
            denoiser_ckpt=denoiser_ckpt,
            hps=hps,
            hps_t2w2v=hps_t2w2v,
            h_sr48=h_sr48,
            hps_denoiser=hps_denoiser,
        )

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    @staticmethod
    def load_text(fp):
        with open(fp, "r") as f:
            filelist = [line.strip() for line in f.readlines()]
        return filelist

    @staticmethod
    def load_checkpoint(filepath, device):
        print(filepath)
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    @staticmethod
    def get_param_num(model):
        num_param = sum(param.numel() for param in model.parameters())
        return num_param

    def intersperse(self, lst, item):
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    def add_blank_token(self, text):
        text_norm = self.intersperse(text, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def load_models(
        self,
        ckpt,
        ckpt_text2w2v,
        ckpt_sr48=None,
        denoiser_ckpt=None,
        hps=None,
        hps_t2w2v=None,
        h_sr48=None,
        hps_denoiser=None,
    ):
        self.net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        ).to(self.device)
        self.net_g.load_state_dict(torch.load(ckpt, map_location=self.device))
        _ = self.net_g.eval()

        self.text2w2v = Text2W2V(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps_t2w2v.model,
        ).to(self.device)
        self.text2w2v.load_state_dict(
            torch.load(ckpt_text2w2v, map_location=self.device)
        )
        self.text2w2v.eval()

        if ckpt_sr48 is not None:
            self.speechsr = SpeechSR48(
                h_sr48.data.n_mel_channels,
                h_sr48.train.segment_size // h_sr48.data.hop_length,
                **h_sr48.model,
            ).to(self.device)
            self.speechsr, _, _, _ = utils.load_checkpoint(
                ckpt_sr48, self.speechsr, None
            )
            self.speechsr.eval()

        if self.load_denoiser and denoiser_ckpt is not None:
            self.denoiser = MPNet(hps_denoiser).to(self.device)
            self.hps_denoiser = hps_denoiser
            state_dict = self.load_checkpoint(denoiser_ckpt, self.device)
            self.denoiser.load_state_dict(state_dict["generator"])
            self.denoiser.eval()

        print("Models loaded.")

    def prompt_audio_to_mel(self, prompt_audio, denoise_ratio=0):
        # Prompt load
        audio, sample_rate = torchaudio.load(prompt_audio)

        # support only single channel
        if audio.shape[0] != 1:
            audio = audio[:1, :]
        # Resampling
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                audio, sample_rate, 16000, resampling_method="kaiser_window"
            )

        # We utilize a hop size of 320 but denoiser uses a hop size of 400 so we utilize a hop size of 1600
        ori_prompt_len = audio.shape[-1]
        p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
        audio = torch.nn.functional.pad(audio, (0, p), mode="constant").data

        if self.load_denoiser and self.denoiser is not None and denoise_ratio != 0:
            with torch.no_grad():
                if ori_prompt_len > 80000:
                    denoised_audio = []
                    for i in range((ori_prompt_len // 80000)):
                        denoised_audio.append(
                            denoise(
                                audio.squeeze(0).to(self.device)[
                                    i * 80000 : (i + 1) * 80000
                                ],
                                self.denoiser,
                                self.hps_denoiser,
                            )
                        )

                    denoised_audio.append(
                        denoise(
                            audio.squeeze(0).to(self.device)[(i + 1) * 80000 :],
                            self.denoiser,
                            self.hps_denoiser,
                        )
                    )
                    denoised_audio = torch.cat(denoised_audio, dim=1)
                else:
                    denoised_audio = denoise(
                        audio.squeeze(0).to(self.device),
                        self.denoiser,
                        self.hps_denoiser,
                    )
            audio = torch.cat(
                [audio.to(self.device), denoised_audio[:, : audio.shape[-1]]], dim=0
            )
        else:
            audio = torch.cat([audio.to(self.device), audio.to(self.device)], dim=0)

        audio = audio[
            :, :ori_prompt_len
        ]  # 20231108 We found that large size of padding decreases a performance so we remove the paddings after denosing.

        if audio.shape[-1] < 48000:
            audio = torch.cat([audio, audio, audio, audio, audio], dim=1)

        mel = self.mel_fn(audio.to(self.device))
        return mel

    def synthesize(
        self,
        text,
        prompt,
        ttv_temperature=0.333,
        vc_temperature=0.333,
        duration_temperature=1.0,
        duration_length=1.0,
        denoise_ratio=0,
        random_seed=1111,
        duration_only=False,
        vec2wav_prompt=None,
    ):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

        text = text_to_sequence(str(text), ["hindi_cleaners2"])

        token = self.add_blank_token(text).unsqueeze(0).to(self.device)
        token_length = torch.LongTensor([token.size(-1)]).to(self.device)

        src_mel = self.prompt_audio_to_mel(prompt, denoise_ratio=denoise_ratio)
        src_length = torch.LongTensor([src_mel.size(2)]).to(self.device)
        src_length2 = torch.cat([src_length, src_length], dim=0)

        if vec2wav_prompt is not None and vec2wav_prompt != prompt:
            trg_mel = self.prompt_audio_to_mel(
                vec2wav_prompt, denoise_ratio=denoise_ratio
            )
            trg_length = torch.LongTensor([src_mel.size(2)]).to(self.device)
            trg_length2 = torch.cat([trg_length, trg_length], dim=0)
        else:
            trg_mel = src_mel
            trg_length = src_length
            trg_length2 = src_length2

        ## TTV (Text --> W2V, F0)
        with torch.no_grad():
            w2v_x, pitch = self.text2w2v.infer_noise_control(
                token,
                token_length,
                src_mel,
                src_length2,
                noise_scale=ttv_temperature,
                noise_scale_w=duration_temperature,
                length_scale=duration_length,
                denoise_ratio=denoise_ratio,
            )

            if duration_only:
                return round(w2v_x.size(2) * 320 / 16000, 3)

            src_length = torch.LongTensor([w2v_x.size(2)]).to(self.device)

            pitch[pitch < torch.log(torch.tensor([55]).to(self.device))] = 0

            ## Hierarchical Speech Synthesizer (W2V, F0 --> 16k Audio)
            converted_audio = self.net_g.voice_conversion_noise_control(
                w2v_x,
                src_length,
                trg_mel,
                trg_length,
                pitch,
                noise_scale=vc_temperature,
                denoise_ratio=denoise_ratio,
            )

            converted_audio = self.speechsr(converted_audio)
            converted_audio = converted_audio.squeeze()
            converted_audio = (
                converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * 0.999
            )
            converted_audio = converted_audio.cpu().numpy().astype("int16")
        return (48000, converted_audio)

    def inference(
        self,
        text="Hello, this is to test the voice.",
        prompt="example/reference_4.wav",
        ttv_temperature=0.333,
        vc_temperature=0.333,
        duration_temperature=1.0,
        duration_length=1.0,
        denoise_ratio=0,
        random_seed=1111,
        duration_only=False,
        save_path="",
        vec2wav_prompt=None,
    ):
        if duration_only:
            duration = self.synthesize(
                text,
                prompt,
                ttv_temperature=ttv_temperature,
                vc_temperature=vc_temperature,
                duration_temperature=duration_temperature,
                duration_length=duration_length,
                denoise_ratio=denoise_ratio,
                random_seed=random_seed,
                duration_only=duration_only,
            )
            return duration

        sr, converted_audio = self.synthesize(
            text,
            prompt,
            ttv_temperature=ttv_temperature,
            vc_temperature=vc_temperature,
            duration_temperature=duration_temperature,
            duration_length=duration_length,
            denoise_ratio=denoise_ratio,
            random_seed=random_seed,
            duration_only=duration_only,
            vec2wav_prompt=vec2wav_prompt,
        )
        if save_path.strip() == "":
            save_path = os.path.join(self.output_dir, f"{uuid4()}.wav")
        write(save_path, sr, converted_audio)
        return save_path


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    syn = HierspeechSynthesizer(
        ckpt="./logs/hierspeechpp_eng_kor/hierspeechpp_v2_ckpt.pth",
        ckpt_text2w2v="./logs/hierspeech_hindi/ttv_hierspeech_hindi_165k.pth",
        ckpt_sr="./speechsr24k/G_340000.pth",
        ckpt_sr48="./speechsr48k/G_100000.pth",
        denoiser_ckpt="denoiser/g_best",
        load_denoiser=True,
        device=device,
        seed=1111,
        output_dir="outputs",
    )

    # Just return duration
    duration = syn.inference(
        text="मैं जिस तरह से सोच रहा था वह यह था कि हम वास्तव में कब जीवित हैं?", prompt="example/reference_1.wav", duration_only=True
    )
    print(f"Duration : {duration} seconds")

    # Return Sample Rate and Audio in Numpy Array
    sr, waveform = syn.synthesize(
        text="मैं जिस तरह से सोच रहा था वह यह था कि हम वास्तव में कब जीवित हैं?", prompt="example/reference_1.wav"
    )
    print(f"Waveform shape, {waveform.shape}")

    # Return audio file path
    path = syn.inference(
        text="मैं जिस तरह से सोच रहा था वह यह था कि हम वास्तव में कब जीवित हैं?", prompt="example/reference_1.wav"
    )
    print(f"Saved in {path}")


    # Return audio file path, different prompts
    path = syn.inference(
        text="मैं जिस तरह से सोच रहा था वह यह था कि हम वास्तव में कब जीवित हैं?", prompt="example/reference_1.wav", vec2wav_prompt="example/reference_3.wav"
    )
    print(f"Saved in {path}")
