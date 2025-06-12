import os
import json
from pathlib import Path
import whisperx
import soundfile as sf
import numpy as np
import re
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys

from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_TOKEN")

print("Start Preprocessing ... ...")

sys.path.append('./CrisperWhisper/')
from utils import adjust_pauses_for_hf_pipeline_output

def generate_session_id():
    session_root = "session_data"
    if not os.path.exists(session_root):
        os.makedirs(session_root)
        return "000001"
    
    existing_ids = [d for d in os.listdir(session_root)
                    if os.path.isdir(os.path.join(session_root, d)) and d.isdigit()]
    if existing_ids:
        new_id = max(int(x) for x in existing_ids) + 1
    else:
        new_id = 1
    return f"{new_id:06d}"

def assign_speakers(segments, diarization_segments):
    speaker_map = {}
    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        max_overlap = 0
        assigned_speaker = "Unknown"
        
        for _, diar in diarization_segments.iterrows():
            speaker = diar["speaker"]
            diar_start = diar["start"]
            diar_end = diar["end"]
            overlap_start = max(segment_start, diar_start)
            overlap_end = min(segment_end, diar_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                assigned_speaker = speaker
        
        speaker_map[segment_start] = assigned_speaker
    return speaker_map

def load_audio_for_split(input_audio_file):

    if input_audio_file.lower().endswith('.mp3'):
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_file(input_audio_file)
        sr = audio_seg.frame_rate
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
        samples = samples / 32768.0
        if audio_seg.channels > 1:
            samples = samples.reshape((-1, audio_seg.channels))
        return samples, sr
    else:
        return sf.read(input_audio_file)

def process_audio_file(input_audio_file, num_speakers, device="cuda"):
    
    print("Loading WhisperX model (English)...")
    model = whisperx.load_model("medium", device, language="en")
    
    audio = whisperx.load_audio(input_audio_file)
    
    print("Transcribing audio with WhisperX...")
    result = model.transcribe(audio)
    
    print("Performing forced alignment with WhisperX...")
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    result_aligned = whisperx.align(result["segments"], alignment_model, metadata, audio, device, return_char_alignments=True)
    
    print("Detecting speakers with WhisperX...")
    diarization_model = whisperx.DiarizationPipeline(use_auth_token=token,
                                                     device=device)
    diarization_segments = diarization_model(audio)
    
    speaker_map = assign_speakers(result_aligned["segments"], diarization_segments)
    for segment in result_aligned["segments"]:
        segment["speaker"] = speaker_map.get(segment["start"], "Unknown")
        segment.pop("chars", None)

    session_id = generate_session_id()
    session_dir = os.path.join("session_data", session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    data, sr = load_audio_for_split(input_audio_file)
    
    for segment in result_aligned["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        speaker = segment["speaker"]
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = data[start_sample:end_sample]
        
        segment_filename = f"{session_id}-{start_time:.2f}-{end_time:.2f}-{speaker}.wav"
        segment_filepath = os.path.join(session_dir, segment_filename)
        sf.write(segment_filepath, segment_audio, sr)
        print(f"Saved segment: {segment_filepath}")
    

    transcript_path = os.path.join(session_dir, f"{session_id}_transcription.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for segment in result_aligned["segments"]:
            f.write(f"[{segment['start']} - {segment['end']}] (Speaker {segment['speaker']}): {segment['text']}\n")
    

    del model
    torch.cuda.empty_cache()
    

    print("Loading CrisperWhisper model...")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    """ Use local Crisper Whisper Model
    local_model_dir = "./CrisperWhisper_local"

    cw_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    cw_model.to(device_str)
    
    processor = AutoProcessor.from_pretrained(local_model_dir)
    
    """
    hf_model_id = "nyrahealth/CrisperWhisper"

    cw_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=token
    )
    cw_model.to(device_str)

    processor = AutoProcessor.from_pretrained(hf_model_id, token=token)

    
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=cw_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,         
        batch_size=4,
        return_timestamps='word',  
        torch_dtype=torch_dtype,
        device=0 if torch.cuda.is_available() else -1,
        generate_kwargs={"language": "en"}
    )
    
    segments_cw = []
    skipped_segments = []
    segment_files = [f for f in os.listdir(session_dir) if f.endswith('.wav')]
    for seg_file in sorted(segment_files):

        match = re.match(r'^(\d+)-(\d+\.\d+)-(\d+\.\d+)-(.+)\.wav$', seg_file)
        if not match:
            continue

        seg_session_id = match.group(1)
        start_time = float(match.group(2))
        end_time = float(match.group(3))
        speaker = match.group(4)
        seg_path = os.path.join(session_dir, seg_file)
        
        print(f"Processing segment with CrisperWhisper: {seg_path}")
        try:
            cw_output = asr_pipeline(seg_path)
            cw_result = adjust_pauses_for_hf_pipeline_output(cw_output)
        except Exception as e:
            print(f"[Warning] CrisperWhisper error, skiped this segment: {seg_path}\nError Message: {e}")
            skipped_segments.append(seg_path)
            continue
        
        text = cw_result.get('text', '').strip()
        if not text:
            print(f"********** No text returned, skiped this segment: {seg_path} **********")
            skipped_segments.append(seg_path)
            continue
        
        chunks = cw_result.get('chunks', [])
        words_info = []
        for i, chunk in enumerate(chunks):
            word_text = chunk['text'].strip()
            if not word_text:
                continue

            chunk_start, chunk_end = chunk['timestamp']
            
            if chunk_start is None:
                if i == 0:
                    chunk_start = 0.0
                else:
                    chunk_start = words_info[-1]['end'] - start_time

            if chunk_end is None:
                if i < len(chunks) - 1:
                    next_chunk_start, _ = chunks[i+1]['timestamp']
                    if next_chunk_start is None:
                        next_chunk_start = chunk_start
                    chunk_end = next_chunk_start
                else:
                    chunk_end = end_time - start_time 

            word_start = round(start_time + chunk_start, 3)
            word_end = round(start_time + chunk_end, 3)
            words_info.append({
                "word": word_text,
                "start": word_start,
                "end": word_end
            })
        
        segment_entry = {
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "speaker": speaker,
            "text": text,
            "words": words_info
        }
        segments_cw.append(segment_entry)
    
    segments_cw = sorted(segments_cw, key=lambda x: x["start"])
    cw_json_path = os.path.join(session_dir, f"{session_id}_transcriptionCW.json")
    with open(cw_json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments_cw}, f, ensure_ascii=False, indent=4)
    print(f"CrisperWhisper transcription saved to: {cw_json_path}")
    
    if skipped_segments:
        skipped_file = os.path.join(session_dir, "skipped_segments.txt")
        with open(skipped_file, "w", encoding="utf-8") as f:
            for s in sorted(skipped_segments):
                f.write(s + "\n")
        print(f"Skipped segments recorded in: {skipped_file}")
    
    return session_id

if __name__ == "__main__":
    session = process_audio_file("/home/easgrad/shuweiho/workspace/volen/SATE_docker_test/input/454.mp3", num_speakers=2, device="cuda")
    print("Processing complete. Session ID:", session)
