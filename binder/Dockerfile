FROM quay.io/jupyter/base-notebook:latest

RUN mamba install -y nodejs=22

USER root
RUN ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') && \
    wget -qO- "https://github.com/NII-cloud-operation/nblibram/releases/latest/download/nblibram_linux_${ARCH}.tar.gz" | tar xz -C /usr/local/bin/ nblibram
USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/jupyter-mynerva

RUN pip install /home/${NB_USER}/jupyter-mynerva
