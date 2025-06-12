import os
import json
import re
import string

# Load Dict
def load_custom_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        return json.load(f)

custom_dict_path = "./syllable_dict_ENNI_refine.json"
custom_dict = load_custom_dict(custom_dict_path)

vowels_phonemes = [
    "iː", "uː", "ɜː", "ɔː", "ɑː",
    "ɪ", "ʊ", "e", "ə", "æ", "ʌ", "ɛ", "ɒ",
    "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ɪə", "eə", "ʊə"
]

def phoneme_type(phoneme):
    return 'V' if phoneme in vowels_phonemes else 'C'

def get_pronunciation_from_dict(word):
    clean_word = word.strip(string.punctuation).lower()
    return custom_dict.get(clean_word, "")

def split_ipa_into_syllables(ipa_str):
    ipa_str = ipa_str.replace("ˈ", ".").replace("ˌ", ".")
    return [s for s in ipa_str.split('.') if s.strip()]

def split_syllable_into_phonemes(syllable):
    vowels_sorted = sorted(vowels_phonemes, key=len, reverse=True)
    phonemes = []
    i = 0
    while i < len(syllable):
        matched = None
        for v in vowels_sorted:
            if syllable[i:i+len(v)] == v:
                matched = v
                break
        if matched:
            phonemes.append(matched)
            i += len(matched)
        else:
            phonemes.append(syllable[i])
            i += 1
    return phonemes

def analyze_word_syllables(word):
    ipa_str = get_pronunciation_from_dict(word)
    if not ipa_str:
        return []
    syllables_ipa = split_ipa_into_syllables(ipa_str)
    syllable_data = []
    for syl in syllables_ipa:
        phs = split_syllable_into_phonemes(syl)
        CV = ''.join(phoneme_type(p) for p in phs)
        syllable_data.append({
            "syllable": ''.join(phs),
            "phonemes": phs,
            "CV_pattern": CV
        })
    return syllable_data

def annotate_syllables(session_id, base_dir="session_data"):
    json_file = os.path.join(base_dir, session_id, f"{session_id}_transcriptionCW.json")
    if not os.path.exists(json_file):
        print(f"[Error] Cannot find file: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for segment in data.get("segments", []):
        text = segment.get("text", "")
        words_info = segment.get("words", [])
        syllables = []

        for idx, word_obj in enumerate(words_info):
            word = word_obj.get("word", "")
            if re.fullmatch(r"\[.*?\]", word):  # 跳过 filler
                continue
            word_syllables = analyze_word_syllables(word)
            for syl in word_syllables:
                syl["word_index"] = idx
            syllables.extend(word_syllables)

        segment["syllables"] = syllables

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Session {session_id} syllable annotation done: {json_file}")
    return data


if __name__ == "__main__":
    annotate_syllables("000030")
