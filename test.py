import os
from preprocess import process_audio_file
from pause import annotate_pauses
from repetition import annotate_repetitions
from syllable import annotate_syllables
from fillerword import annotate_fillerwords
from mispronunciation import annotate_mispronunciation

from feature_extraction import feature_extraction


from annotation import annotate_transcript

def main():

    input_audio_file = "/home/easgrad/shuweiho/workspace/volen/SATE_docker_test/input/454.mp3"                 
    device = "cuda"                            
    pause_threshold = 0.3                       

    print("Start init...")
    
    session_id = process_audio_file(input_audio_file, num_speakers=2, device=device)

    # annotation
    annotate_pauses(session_id, pause_threshold)
    annotate_repetitions(session_id)
    annotate_syllables(session_id)
    annotate_fillerwords(session_id)
    # annotate_mispronunciation(session_id, api_url="http://localhost:8080")
    
    # feature extraction
    # feature_extraction(session_id)

    # transcription generation
    output_annotation = annotate_transcript(session_id)
    print(f"Done: {output_annotation}")

if __name__ == "__main__":
    main()
