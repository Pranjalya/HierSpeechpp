import requests

URL = "http://127.0.0.1:5000/get_duration"

payload = {
    'text': 'Hello, testing the voice.',
    'denoise_ratio': 0,
    'duration_only': True
}

# Define the audio file to upload
audio_file_path = '/mnt/d/3_100.wav'  # Update with the actual path to your audio file
files = {'audio': ('3_100.wav', open(audio_file_path, 'rb'))}

# Make the POST request to the API endpoint
response = requests.post(URL, data=payload, files=files)

# Check the response status code
if response.status_code == 200:
    # Save the processed audio response to a file
    with open('processed_audio.wav', 'wb') as output_file:
        output_file.write(response.content)
    print("Processed audio saved as 'processed_audio.wav'")
else:
    print(f"Error: {response.status_code} - {response.text}")