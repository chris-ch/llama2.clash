FROM debian:bookworm-slim

ARG GHCUP_DWN_URL=https://downloads.haskell.org/~ghcup/x86_64-linux-ghcup

ARG PATH
RUN test -n ${PATH}
ENV PATH=${PATH}
ENV LANG=C.UTF-8
ENV BOOTSTRAP_HASKELL_NONINTERACTIVE=1

RUN \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl \
        libnuma-dev \
        zlib1g-dev \
        libgmp-dev \
        libgmp10 \
        git \
        wget \
        lsb-release \
        software-properties-common \
        gnupg2 \
        apt-transport-https \
        gcc \
        autoconf \
        automake \
        libffi-dev \
        libffi8 \
        libgmp-dev \
        libgmp10 \
        libncurses-dev \
        libncurses5 \
        libtinfo5 \
        libblas3 \
        liblapack3 \
        liblapack-dev \
        libblas-dev \
        llvm \
        xz-utils \
        ssh \
        vim \
        time \
        build-essential

# install gpg keys
RUN \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys 7D1E8AFD1D4A16D71FADA2F2CCC85C0E40C06A8C && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys FE5AB6C91FEA597C3B31180B73EDE9E8CFBAEF01 && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys 88B57FCF7DB53B4DB3BFA4B1588764FBE22D19C4 && \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys EAF2A9A722C0C96F2B431CA511AAD8CEDEE0CAEF

# install ghcup
RUN \
    curl ${GHCUP_DWN_URL} > /usr/bin/ghcup && \
    chmod +x /usr/bin/ghcup && \
    ghcup config set gpg-setting GPGStrict

ARG VERSION_GHC=9.2.8
ARG VERSION_CABAL=3.12.1.0

ARG USER_NAME=haskell
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --no-log-init --create-home -m ${USER_NAME} -s /usr/bin/bash \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get install -y sudo \
    && echo ${USER_NAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}

RUN groupadd docker && \
    usermod -aG docker ${USER_NAME}

USER ${USER_NAME}

WORKDIR /home/${USER_NAME}

# install GHC and cabal
RUN \
    ghcup -v install ghc --force ${VERSION_GHC} && \
    ghcup -v install cabal --force ${VERSION_CABAL} && \
    ghcup set ghc ${VERSION_GHC} && \
    ghcup install hls

RUN cabal install clash-ghc

RUN /bin/echo -e "\nexport PATH=$PATH:/home/${USER_NAME}/.ghcup/bin:/home/${USER_NAME}/.local/bin/\n" >> /home/${USER_NAME}/.bashrc
