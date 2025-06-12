import os
import tempfile
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from preprocess import process_audio_file
from pause import annotate_pauses
from repetition import annotate_repetitions
from syllable import annotate_syllables
from fillerword import annotate_fillerwords

from annotation import annotate_transcript



app = Flask(__name__)

@app.route('/process_old', methods=['POST'])
def process_audio_old():
    data = request.get_json()
    if not data or 'input_audio_file' not in data:
        return jsonify({'error': 'Missing input_audio_file parameter'}), 400

    input_audio_file = data['input_audio_file']
    device = data.get('device', 'cuda')
    pause_threshold = data.get('pause_threshold', 0.5)
    num_speakers = data.get('num_speakers', 2)

    app.logger.info(f"Processing audio file: {input_audio_file}")

    session_id = process_audio_file(input_audio_file, num_speakers=num_speakers, device=device)
    annotate_pauses(session_id, pause_threshold)
    annotate_repetitions(session_id)
    annotate_syllables(session_id)
    annotate_fillerwords(session_id)

    output_annotation = annotate_transcript(session_id)

    result = {
        'session_id': session_id,
        'annotation_result': output_annotation
    }
    return jsonify(result), 200



@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'Missing audio file '}), 400
    audio_file = request.files['audio_file']
    filename = secure_filename(audio_file.filename)
    
    suffix = os.path.splitext(filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    device = request.form.get('device', 'cuda')
    pause_threshold = float(request.form.get('pause_threshold', 0.5))
    num_speakers = int(request.form.get('num_speakers', 2))

    app.logger.info(f"Processing uploaded audio: {audio_path}")

    session_id = process_audio_file(audio_path, num_speakers=num_speakers, device=device)
    annotate_pauses(session_id, pause_threshold)
    annotate_repetitions(session_id)
    # annotate_syllables(session_id)
    annotate_fillerwords(session_id)
    # annotate_transcript(session_id)


    json_path = f"session_data/{session_id}/{session_id}_transcriptionCW.json"
    if not os.path.isfile(json_path):
        return jsonify({'error': f"Annotation file {json_path} not found"}), 500

    with open(json_path, 'r', encoding='utf-8') as f:
        transcription = json.load(f)


    try:
        os.remove(audio_path)
    except OSError:
        pass

    return jsonify(transcription), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
