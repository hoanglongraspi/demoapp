import os
import json
import re
from collections import Counter

def target_speaker(segments):
    speakers = [seg.get("speaker", "UNKNOWN") for seg in segments]
    most_common = Counter(speakers).most_common(1)
    return most_common[0][0] if most_common else "UNKNOWN"

def pause_feature(session_id, base_dir="session_data"):

    json_file = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    feature_file = os.path.join(base_dir, session_id, f"{session_id}_feature.json")

    if not os.path.exists(json_file):
        print(f"[Error] transcriptionCW file not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    tgt_speaker = target_speaker(segments)
    tgt_segments = [seg for seg in segments if seg.get("speaker") == tgt_speaker]

    total_words = 0
    total_duration = 0.0
    all_pauses = []

    for seg in tgt_segments:
        words = seg.get("words", [])
        total_words += len(words)
        duration = seg.get("end", 0) - seg.get("start", 0)
        total_duration += duration
        pauses = seg.get("pauses", [])
        all_pauses.extend(pauses)

    total_pauses = len(all_pauses)
    pause_durations = [p["duration"] for p in all_pauses]
    pause_total_duration = sum(pause_durations)
    avg_pause_duration = sum(pause_durations) / total_pauses if total_pauses > 0 else 0
    longest_pause = max(pause_durations) if pause_durations else 0

    pause_density_1 = (total_pauses / total_words * 100) if total_words > 0 else 0
    pause_density_2 = (total_pauses / total_duration * 60) if total_duration > 0 else 0
    pause_proportion = (pause_total_duration / total_duration) * 100 if total_duration > 0 else 0

    pause_feat = {
        "Pause Density I": round(pause_density_1, 3),
        "Pause Density II": round(pause_density_2, 3),
        "Pause Proportion": round(pause_proportion, 3),
        "Average Pause Duration": round(avg_pause_duration, 3),
        "Longest Pause Duration": round(longest_pause, 3)
    }

    if os.path.exists(feature_file):
        with open(feature_file, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
    else:
        feature_data = {}

    feature_data.setdefault("pause", []).append(pause_feat)

    with open(feature_file, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4, ensure_ascii=False)

    print(f"[Done] Pause features written to {feature_file}")



def get_syllable_weight(cv_pattern):
    weight1_patterns = {'CV', 'CVC', 'VC', 'V'}
    weight2_patterns = {'VCC', 'CVCC', 'CCV', 'CCVC'}
    if cv_pattern in weight1_patterns:
        return 1
    elif cv_pattern in weight2_patterns:
        return 10
    else:
        return 100
        

def syllable_feature(session_id, base_dir="session_data"):
    json_file = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    feature_file = os.path.join(base_dir, session_id, f"{session_id}_feature.json")

    if not os.path.exists(json_file):
        print(f"[Error] transcriptionCW file not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    tgt_speaker = target_speaker(segments)
    tgt_segments = [seg for seg in segments if seg.get("speaker") == tgt_speaker]

    total_duration = sum(seg["end"] - seg["start"] for seg in tgt_segments)
    syllable_types = set()
    total_syllables = 0
    total_weight = 0
    max_syllable_len = 0
    word_syllable_count = {}
    word_counter = 0 

    for seg in tgt_segments:
        syllables = seg.get("syllables", [])
        for syl in syllables:
            cv = syl.get("CV_pattern", "")
            phonemes = syl.get("phonemes", [])
            syllable_types.add(cv)
            total_syllables += 1
            total_weight += get_syllable_weight(cv)
            max_syllable_len = max(max_syllable_len, len(phonemes))
            
            local_idx = syl.get("word_index")
            global_idx = word_counter + local_idx
            word_syllable_count[global_idx] = word_syllable_count.get(global_idx, 0) + 1
        word_counter += len(seg.get("words", []))

    all_words = []
    for seg in tgt_segments:
        all_words.extend(seg.get("words", []))
    total_words = len(all_words)

    multi_syll_2 = sum(1 for c in word_syllable_count.values() if c >= 2)
    multi_syll_3 = sum(1 for c in word_syllable_count.values() if c >= 3)

    feat = {
        "Number of Syllable Types": len(syllable_types),
        "Syllable Complexity Index": round(total_weight / total_syllables, 3) if total_syllables > 0 else 0,
        "Average Syllable Rate": round(total_syllables / total_duration, 3) if total_duration > 0 else 0,
        "Longest Syllable Length": max_syllable_len,
        "Proportion of Multisyllabic Words I": round((multi_syll_2 / total_words) * 100, 3) if total_words > 0 else 0,
        "Proportion of Multisyllabic Words II": round((multi_syll_3 / total_words) * 100, 3) if total_words > 0 else 0
    }

    if os.path.exists(feature_file):
        with open(feature_file, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
    else:
        feature_data = {}

    feature_data.setdefault("syllable", []).append(feat)

    with open(feature_file, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4, ensure_ascii=False)

    print(f"[Done] Syllable features written to {feature_file}")



def get_rep_weight(length):
    if length == 1:
        return 1
    elif length == 2:
        return 5
    elif length == 3:
        return 10
    elif length == 4:
        return 20
    else:
        return 40


def repetition_feature(session_id, base_dir="session_data"):
    json_file = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    feature_file = os.path.join(base_dir, session_id, f"{session_id}_feature.json")

    if not os.path.exists(json_file):
        print(f"[Error] transcriptionCW file not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    tgt_speaker = target_speaker(segments)
    tgt_segments = [seg for seg in segments if seg.get("speaker") == tgt_speaker]

    total_duration = sum(seg["end"] - seg["start"] for seg in tgt_segments)
    total_words = sum(len(seg.get("words", [])) for seg in tgt_segments)

    rep_count = 0
    rep_weights = 0
    rep_lengths = []
    rep_durations = []

    for seg in tgt_segments:
        words = seg.get("words", [])
        repetitions = seg.get("repetitions", [])
        for rep in repetitions:
            indices = rep.get("words", [])
            if not indices:
                continue
            length = len(indices)
            rep_lengths.append(length)
            rep_weights += get_rep_weight(length)
            rep_count += 1

            # compute duration based on word timestamps
            start_idx = indices[0]
            end_idx = indices[-1]
            if 0 <= start_idx < len(words) and 0 <= end_idx < len(words):
                start_time = words[start_idx]["start"]
                end_time = words[end_idx]["end"]
                rep_durations.append(end_time - start_time)

    total_rep_time = sum(rep_durations)
    longest_rep = max(rep_lengths) if rep_lengths else 0
    avg_rep_len = sum(rep_lengths) / len(rep_lengths) if rep_lengths else 0

    rep_feat = {
        "Word Repetition Index": round(rep_weights / rep_count, 3) if rep_count > 0 else 0,
        "Repetition Density I": round(rep_count / total_words * 100, 3) if total_words > 0 else 0,
        "Repetition Density II": round(rep_count / total_duration * 60, 3) if total_duration > 0 else 0,
        "Repetition Proportion": round((total_rep_time / total_duration) * 100, 3) if total_duration > 0 else 0,
        "Longest Repetition Length": longest_rep,
        "Average Repetition Length": round(avg_rep_len, 3)
    }

    if os.path.exists(feature_file):
        with open(feature_file, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
    else:
        feature_data = {}

    feature_data.setdefault("repetition", []).append(rep_feat)

    with open(feature_file, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4, ensure_ascii=False)

    print(f"[Done] Repetition features written to {feature_file}")



def fillerword_feature(session_id, base_dir="session_data"):
    json_file = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    feature_file = os.path.join(base_dir, session_id, f"{session_id}_feature.json")

    if not os.path.exists(json_file):
        print(f"[Error] transcriptionCW file not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    tgt_speaker = target_speaker(segments)
    tgt_segments = [seg for seg in segments if seg.get("speaker") == tgt_speaker]

    total_duration = sum(seg["end"] - seg["start"] for seg in tgt_segments)
    total_words = sum(len(seg.get("words", [])) for seg in tgt_segments)

    filler_total = 0
    filler_durations = []
    filler_intervals = []

    for seg in tgt_segments:
        fillers = seg.get("fillerwords", [])
        words = seg.get("words", [])

        filler_total += len(fillers)
        filler_durations += [fw["duration"] for fw in fillers]

        if len(fillers) >= 2:
            indices = []
            for fw in fillers:
                for i, w in enumerate(words):
                    if abs(w.get("start", 0) - fw["start"]) < 0.01:
                        indices.append(i)
                        break
            indices.sort()
            intervals = [indices[i+1] - indices[i] - 1 for i in range(len(indices)-1)]
            if intervals:
                filler_intervals.append(sum(intervals) / len(intervals))

    avg_duration = sum(filler_durations) / len(filler_durations) if filler_durations else 0
    longest_duration = max(filler_durations) if filler_durations else 0
    avg_interval = sum(filler_intervals) / len(filler_intervals) if filler_intervals else 0

    feat = {
        "Filler Word Density I": round(filler_total / total_words * 100, 3) if total_words > 0 else 0,
        "Filler Word Density II": round(filler_total / total_duration * 60, 3) if total_duration > 0 else 0,
        "Filler Word Proportion": round((sum(filler_durations) / total_duration) * 100, 3) if total_duration > 0 else 0,
        "Average Filler Word Duration": round(avg_duration, 3),
        "Longest Filler Word Duration": round(longest_duration, 3),
        "Average Filler Word Interval": round(avg_interval, 3)
    }

    if os.path.exists(feature_file):
        with open(feature_file, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
    else:
        feature_data = {}

    feature_data.setdefault("fillerword", []).append(feat)

    with open(feature_file, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4, ensure_ascii=False)

    print(f"[Done] Filler word features written to {feature_file}")


def plm_feature(session_id, base_dir="session_data"):


    json_file = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    feature_file = os.path.join(base_dir, session_id, f"{session_id}_feature.json")

    if not os.path.exists(json_file):
        print(f"[Error] transcriptionCW file not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    tgt_speaker = target_speaker(segments)
    tgt_segments = [seg for seg in segments if seg.get("speaker") == tgt_speaker]

    sequence = ''.join(seg.get("mispronunciation", "") for seg in tgt_segments)
    sequence = re.sub(r"[^CE]", "", sequence.upper())

    n = len(sequence)
    if n == 0:
        feat = {
            "Mispronunciation Density": 0,
            "Normalized Transition Count": 0,
            "Average Common Correct": 0,
            "Average Common Error": 0,
            "Longest Common Correct": 0,
            "Longest Common Error": 0
        }
    else:
        transitions = sum(1 for i in range(1, n) if sequence[i] != sequence[i - 1])
        MPD = sum(1 for ch in sequence if ch == 'E') / n
        NTC = transitions / n

        def run_lengths(s, ch):
            return [len(g) for g in re.findall(f"{ch}+", s)]

        c_runs = run_lengths(sequence, 'C')
        e_runs = run_lengths(sequence, 'E')

        ACC = sum(c_runs) / len(c_runs) if c_runs else 0
        ACE = sum(e_runs) / len(e_runs) if e_runs else 0
        LCC = max(c_runs) if c_runs else 0
        LCE = max(e_runs) if e_runs else 0

        feat = {
            "Mispronunciation Density": round(MPD, 3),
            "Normalized Transition Count": round(NTC, 3),
            "Average Common Correct": round(ACC, 3),
            "Average Common Error": round(ACE, 3),
            "Longest Common Correct": LCC,
            "Longest Common Error": LCE
        }

    if os.path.exists(feature_file):
        with open(feature_file, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
    else:
        feature_data = {}

    feature_data.setdefault("plm", []).append(feat)

    with open(feature_file, "w", encoding="utf-8") as f:
        json.dump(feature_data, f, indent=4, ensure_ascii=False)

    print(f"[Done] PLM features written to {feature_file}")



def feature_extraction(session_id, base_dir="session_data"):

    feature_file = os.path.join(base_dir, session_id, f"{session_id}_feature.json")

    if not os.path.exists(feature_file):
        with open(feature_file, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)

    pause_feature(session_id, base_dir=base_dir)
    syllable_feature(session_id, base_dir=base_dir)
    repetition_feature(session_id, base_dir=base_dir)
    fillerword_feature(session_id, base_dir=base_dir)
    plm_feature(session_id, base_dir=base_dir)

    print(f"[All Analysis Done] Feature extraction complete for session {session_id}")


if __name__ == "__main__":
    feature_extraction("000030")
