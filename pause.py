import os
import json

def annotate_pauses(session_id, threshold, base_dir="session_data"):

    session_dir = os.path.join(base_dir, session_id)
    json_file = os.path.join(session_dir, f"{session_id}_transcriptionCW.json")
    
    if not os.path.exists(json_file):
        print(f"Error: could not finf {json_file}")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    for segment in segments:
        words = segment.get("words", [])
        if "pauses" in segment:
            del segment["pauses"]
            
        pauses = []
        if words and len(words) > 1:
            for i in range(1, len(words)):
                prev_word = words[i - 1]
                current_word = words[i]
                gap = current_word["start"] - prev_word["end"]
                if gap > threshold:
                    pause_info = {
                        "start": round(prev_word["end"], 3),
                        "end": round(current_word["start"], 3),
                        "duration": round(gap, 3)
                    }
                    pauses.append(pause_info)
        segment["pauses"] = pauses
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Session {session_id} pause annotation done: {json_file}")
    return data

if __name__ == "__main__":
    annotated_data = annotate_pauses("000030", 0.1)
