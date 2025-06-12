This is the SATE MVP, integrate the all the pipelines into one framework.
Contain the main entrance, build for docker.

main.py:
Input: Entire audio file
Output: Transcription with annotation

Preprocess:
Segmentation + speaker diarization -> crisper whisper transcriptions for each segmentation


P.S. Should keep transcript consist in each pipelines.



IMAGE CREATION:

docker build -t sate_0.11 .


(New) HOW TO USE after image created:

docker run --gpus all -it --rm \
  -v /home/easgrad/shuweiho/workspace/volen/SATE_docker_test/input:/sate/input \
  -v /home/easgrad/shuweiho/workspace/volen/SATE_docker_test/session_data:/sate/session_data \
  -p 7860:7860 \
  sate_0.11


curl -X POST http://localhost:7860/process \
  -F "audio_file=@/home/easgrad/shuweiho/workspace/volen/SATE_docker_test/input/454.mp3" \
  -F "device=cuda" \
  -F "pause_threshold=0.25"




(Old - don't follow it) HOW TO USE after image created:

docker run --gpus all -it --rm \
  -v /home/easgrad/shuweiho/workspace/volen/SATE_docker_test/input:/sate/input \
  -v /home/easgrad/shuweiho/workspace/volen/SATE_docker_test/session_data:/sate/session_data \
  -p 5000:5000 \
  sate_0.10


curl -X POST http://localhost:5000/process \
     -H "Content-Type: application/json" \
     -d '{
           "input_audio_file": "/sate/input/454.mp3",
           "device": "cuda",
           "pause_threshold": 0.5
         }'


Test on HF space:

curl -X POST https://Sven33-SATE.hf.space/process   -F "audio_file=@/home/easgrad/shuweiho/workspace/volen/SATE_docker_test/input/454.mp3"   -F "device=cuda"   -F "pause_threshold=0.25"