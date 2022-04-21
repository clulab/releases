FROM nvidia/cuda:11.6.0-base-ubuntu20.04
#FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

LABEL maintainer="Gus Hahn-Powell <gus@parsertongue.org>"
LABEL description="Image definition for Conda-based Python project."

# FIXME: consider moving to a requirements.txt and no conda

ARG PLATFORM=amd64
ARG PYTHON_VERSION=3.9
ARG CONDA_DIR=/opt/conda
ARG USERNAME=odinsynth
ARG USERID=1000
ARG ENVFILE=requirements.txt

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \ 
      wget \
      unzip \ 
      bzip2 \
      sudo \
      build-essential \
      ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH

# arm64 or amd64 conda?
RUN if [ "$PLATFORM" = "arm64" ]; then wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh ; else https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh ; fi

# update path
RUN echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME \
    && chown $USERNAME $CONDA_DIR -R \
    && adduser $USERNAME sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

# set default python version
RUN conda install python=${PYTHON_VERSION}

# Install mamba
RUN conda install -y mamba -c conda-forge

# add source
ADD . .

#RUN mamba env update --file environments/$ENVFILE && conda clean -tipy

RUN pip install -r environments/$ENVFILE

RUN conda init bash
RUN echo "conda activate base" >> /home/$USERNAME/.bashrc

RUN sudo chmod 777 bash_scripts/*

# FIXME: what port is this supposed to run on?
CMD [ "./home/$USERNAME/bash_scripts/start_server.sh", "/data/checkpoints/best.ckpt"]