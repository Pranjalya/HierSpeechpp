import os
import torch
import gradio as gr
from inference_class import HierspeechSynthesizer

def inference(text, prompt, vec2wav_prompt, duration_only, ttv_temperature, vc_temperature, duration_temperature, duration_length, denoise_ratio, random_seed):
    if duration_only:
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
            vec2wav_prompt=vec2wav_prompt
        )
        return duration, None
    else:
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
            vec2wav_prompt=vec2wav_prompt
        )
        return "", (sr, waveform)


text = gr.Textbox(label="Text", required=True)
prompt = gr.Audio(type="filepath", label="Prompt", required=True)
vec2wav_prompt = gr.Audio(type="filepath", label="Vector to Waveform Prompt (Leave it blank if we use same prompt as above)")
duration_only = gr.Checkbox(value=True, label="Return only duration")
ttv_temperature = gr.Slider(0, 1, 0.333, label="Text to Vector Temperature")
vc_temperature = gr.Slider(0, 1, 0.333, label="Voice Conversion Temperature")
duration_temperature = gr.Slider(0, 1, 1.0, label="Duration Temperature")
duration_length = gr.Slider(0.5, 2, 1.0, label="Duration length factor")
denoise_ratio = gr.Slider(0, 1, 0, label="Denoise Ratio")
random_seed = gr.Slider(0, 9999, 1111, label="Random Seed")

duration_str = gr.Textbox(label="Duration")
generated_audio = gr.Audio(label="Generated audio", type="numpy")

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)

interface = gr.Interface(
    inference,
    [text, prompt, vec2wav_prompt, duration_only, ttv_temperature, vc_temperature, duration_temperature, duration_length, denoise_ratio, random_seed],
    [duration_str, generated_audio],
    title="Hierspeech Deepsync Demo Gradio",
    theme=theme
)

if __name__=="__main__":
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
    interface.launch(share=True)