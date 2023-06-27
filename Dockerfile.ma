FROM nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /

COPY requirements.txt ./

RUN python -m pip install --upgrade pip
RUN pip install wheel && \
    pip install -r requirements.txt && \
    pip install gradio

COPY scripts/generate/model.py /
COPY scripts/generate/handler.py /
COPY scripts/generate/template.json /

RUN python model.py

COPY scripts/generate/medalpaca_i.py /


CMD ["python", "medalpaca_i.py"]
