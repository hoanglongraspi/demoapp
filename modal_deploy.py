import modal, subprocess, signal

app = modal.App("sate-api")

image = modal.Image.from_dockerfile("Dockerfile")

@app.function(
    image=image,
    min_containers=1,
    timeout=1000,
    gpu="A10G",
)

@modal.web_server(7860)
def run_server():
    subprocess.Popen([
        "conda", "run", "--no-capture-output", "-n", "SATE",
        "python", "main_socket.py", "--host", "0.0.0.0", "--port", "7860"
    ])
    signal.pause()
