FROM nvidia/cuda:12.0.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    espeak-ng \
    curl \
    ffmpeg

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech/
RUN mkdir /FlaskTtS
RUN ln -s /StyleTTS2-LJSpeech/Models /FlaskTtS/Models
ADD Utils /FlaskTtS/Utils
WORKDIR /FlaskTtS

RUN pip3 install --upgrade pip
RUN pip3 install nltk
RUN python3 -c "import nltk;nltk.download('punkt_tab')"

ADD requirements.txt /FlaskTtS/requirements.txt
RUN pip3 install -r requirements.txt

COPY entrypoint.sh /FlaskTtS/
RUN chmod +x /FlaskTtS/entrypoint.sh

ADD pyproject.toml /FlaskTtS/pyproject.toml
ADD flasktts/ /FlaskTtS/flasktts

RUN pip3 install -e .

CMD ["/FlaskTtS/entrypoint.sh"]
