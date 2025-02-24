FROM nvidia/cuda:12.0.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ffmpeg \
    cmake
RUN python3 -m pip install --upgrade pip
RUN git clone https://github.com/espeak-ng/espeak-ng/
RUN mkdir espeak-ng/build
WORKDIR /espeak-ng/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON
RUN make
RUN make install
RUN ldconfig
WORKDIR /
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

RUN pip3 install -I git+https://github.com/thewh1teagle/phonemizer-fork.git@dev

COPY entrypoint.sh /FlaskTtS/
RUN chmod +x /FlaskTtS/entrypoint.sh

ADD pyproject.toml /FlaskTtS/pyproject.toml
ADD flasktts/ /FlaskTtS/flasktts

RUN pip3 install -e .

CMD ["/FlaskTtS/entrypoint.sh"]
