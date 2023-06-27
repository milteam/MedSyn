FROM nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /scripts/generate

COPY requirements.txt ./

RUN python -m pip install --upgrade pip
RUN pip install wheel && \
    pip install -r requirements.txt && \
    pip install gradio

COPY scripts ./

RUN python generate/model.py

CMD ["python", "generate/medalpaca_i.py"]
