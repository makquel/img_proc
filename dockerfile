# # https://pythonspeed.com/articles/activate-conda-dockerfile/
FROM gcr.io/google-appengine/python

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev postgresql build-essential libffi-dev python3-pip python3-dev apt-utils wget
# # https://stackoverflow.com/questions/58269375/how-to-install-packages-with-miniconda-in-dockerfile
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo PATH="/root/miniconda3/bin":$PATH >> .bashrc \
    && exec bash \
    && conda --version

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]
RUN conda --version

WORKDIR /app

# # Create the environment:
ADD ./environment.yml .
RUN conda env create -f environment.yml
# # Initialize conda in bash config files:
RUN conda init bash
# # Activate the environment
RUN echo "conda activate img_proc" > ~/.bashrc
# # The code to run when container is started:
ADD ./main.py .
# # For production purpose
ENTRYPOINT ["python3", "main.py"]
# # For debugging purpose
# ENTRYPOINT ["/bin/bash", "-c", "while sleep 1000; do :; done"]
