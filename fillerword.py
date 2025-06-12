import os
import json
import re

def annotate_fillerwords(session_id, base_dir="session_data"):
    session_dir = os.path.join(base_dir, session_id)
    json_file = os.path.join(session_dir, f"{session_id}_transcriptionCW.json")

    if not os.path.exists(json_file):
        print(f"[Error] File not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for segment in data.get("segments", []):
        words = segment.get("words", [])
        fillerwords = []
        for w in words:
            word_content = w.get("word", "").strip()
            if re.fullmatch(r"\[.*?\]", word_content):
                fillerwords.append({
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "content": word_content,
                    "duration": round(w.get("end", 0) - w.get("start", 0), 3)
                })
        segment["fillerwords"] = fillerwords

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Session {session_id} fillerword annotation done: {json_file}")
    return data

if __name__ == "__main__":
    annotate_fillerwords("000003")
