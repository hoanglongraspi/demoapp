FROM continuumio/miniconda3:24.11.1-0

RUN useradd -m -u 1000 user
WORKDIR /app

COPY . .

# This is only for building docker in huggingface spaces
ENV HF_HOME=/data/.huggingface

RUN conda env create -f environment_sate_0.11.yml

RUN mkdir -p /app/session_data && chown -R user:user /app/session_data

EXPOSE 7860

CMD ["conda", "run", "--no-capture-output", "-n", "SATE", "python", "main_socket.py", "--port", "7860", "--host", "0.0.0.0"]