FROM nvcr.io/nvidia/pytorch:23.01-py3
LABEL org.opencontainers.image.authors="soulteary@gmail.com"
WORKDIR /scripts/generate

COPY requirements.txt ./

RUN python -m pip install --upgrade pip
RUN pip install wheel && \
    python setup.py bdist_wheel && \
    pip install -r requirements.txt && \
    pip install -e . && \
    pip install gradio

COPY scripts ./

CMD ["python", "medalpaca_i.py"]