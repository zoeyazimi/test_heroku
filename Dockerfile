FROM continuumio/miniconda3:4.12.0

RUN apt update && \
    apt install sudo


COPY src/ /home/
WORKDIR /home/
RUN conda config --add channels conda-forge && \
    conda install -y mamba libarchive=3.5.2 && \
    mamba env update --name base --file env_from_docker.yaml

