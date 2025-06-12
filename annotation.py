import os
import json

def annotate_transcript(session_id, base_dir="session_data"):

    session_dir = os.path.join(base_dir, session_id)
    json_file = os.path.join(session_dir, f"{session_id}_transcriptionCW.json")
    output_file = os.path.join(session_dir, "annotation_result.txt")
    
    if not os.path.exists(json_file):
        print(f"Error: could not find {json_file}")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    annotated_lines = []
    
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        words = segment.get("words", [])
        n = len(words)
        
        pause_map = {}
        for pause in segment.get("pauses", []):
            pause_start = pause.get("start")
            duration = pause.get("duration")
            for idx, token in enumerate(words):
                if abs(token.get("end", 0) - pause_start) < 0.01:
                    pause_map.setdefault(idx, []).append(f"({duration})")
                    break
        
        rep_map = {}
        for rep in segment.get("repetitions", []):
            indices = rep.get("words", [])
            if indices:
                start_idx = indices[0]
                end_idx = indices[-1]
                rep_content = rep.get("content", "")
                rep_map[start_idx] = (end_idx, rep_content)
        
        annotated_tokens = []
        i = 0
        while i < n:
            if i in rep_map:
                rep_end, rep_content = rep_map[i]
                rep_str = f"<{rep_content}> [/]"
                annotated_tokens.append(rep_str)
                if rep_end in pause_map:
                    for pause_marker in pause_map[rep_end]:
                        annotated_tokens.append(pause_marker)
                i = rep_end + 1
            else:
                token_word = words[i].get("word", "")
                annotated_tokens.append(token_word)
                if i in pause_map:
                    for pause_marker in pause_map[i]:
                        annotated_tokens.append(pause_marker)
                i += 1
        
        # join all transcript
        transcript = " ".join(annotated_tokens)
        # gen *SPEAKER:\ttranscript
        line = f"*{speaker}\t{transcript}"
        annotated_lines.append(line)
    
    # write annotation_result.txt
    with open(output_file, "w", encoding="utf-8") as f:
        for line in annotated_lines:
            f.write(line + "\n")
    
    print(f"Annotation done in {output_file}")
    return output_file

if __name__ == "__main__":

    annotate_transcript("000030")
