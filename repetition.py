import os
import json

def annotate_repetitions(session_id, base_dir="session_data"):

    session_dir = os.path.join(base_dir, session_id)
    json_file = os.path.join(session_dir, f"{session_id}_transcriptionCW.json")
    
    if not os.path.exists(json_file):
        print(f"Error: could not find {json_file}")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    for segment in segments:
        if "repetitions" in segment:
            del segment["repetitions"]
            
        words_list = segment.get("words", [])
        tokens = [w.get("word", "") for w in words_list]
        reps = []
        i = 0
        n = len(tokens)
        while i < n:
            found = False
            maxL = (n - i) // 2
            for L in range(maxL, 0, -1):
                if tokens[i:i+L] == tokens[i+L:i+2*L]:
                    count = 2
                    while i + count * L <= n and tokens[i:i+L] == tokens[i+(count-1)*L:i+count*L]:
                        count += 1
                    count -= 1 
                    
                    rep_count = count - 1
                    rep_obj = {
                        "content": " ".join(tokens[i:i+L] * rep_count),
                        "words": list(range(i, i + rep_count * L)),
                        "mark_location": i + rep_count * L - 1
                    }
                    reps.append(rep_obj)
                    i += count * L 
                    found = True
                    break
            if not found:
                i += 1
        segment["repetitions"] = reps
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Session {session_id} repetition annotation done: {json_file}")
    return data

if __name__ == "__main__":
    annotate_repetitions("000030")
