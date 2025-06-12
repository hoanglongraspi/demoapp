import os
import json
import base64
import requests

def wav_to_base64_url(wav_path):
    with open(wav_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode()
    return f"data:audio/wav;base64,{b64_audio}"

def is_api_available(api_url):
    try:
        response = requests.get(api_url)
        return response.status_code < 500
    except Exception as e:
        print("[Error] Cannot reach API:", e)
        return False

def annotate_mispronunciation(session_id, api_url="http://localhost:8080", base_dir="session_data"):
    json_path = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    if not os.path.exists(json_path):
        print(f"[Error] File not found: {json_path}")
        return

    if not is_api_available(api_url):
        print(f"[Error] API not available at {api_url}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        speaker = segment["speaker"]
        filename = f"{session_id}-{start:.2f}-{end:.2f}-{speaker}.wav"
        filepath = os.path.join(base_dir, session_id, filename)
        if not os.path.exists(filepath):
            print(f"[Warning] Audio file missing: {filename}")
            continue

        audio_url = wav_to_base64_url(filepath)
        try:
            resp = requests.post(
                f"{api_url}/vocallens/api/analyze",
                headers={"Content-Type": "application/json"},
                data=json.dumps([audio_url])
            )
            if resp.status_code == 200:
                result = resp.json()
                segment["mispronunciation"] = result.get("ce", "")
            else:
                print(f"[Warning] API error {resp.status_code} for {filename}")
                segment["mispronunciation"] = ""
        except Exception as e:
            print(f"[Error] Exception during API call for {filename}:", e)
            segment["mispronunciation"] = ""

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Session {session_id} mispronunciation annotation done: {json_path}")
    return data

if __name__ == "__main__":
    annotate_mispronunciation("000030")
