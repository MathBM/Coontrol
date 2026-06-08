FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    pkg-config \
    libglib2.0-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xfixes0 \
    libxcb-cursor0 \
    libxcb-shape0 \
    libxcb-render0 \
    libxcb-shm0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgomp1 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

WORKDIR /workspaces

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV QT_DEBUG_PLUGINS=1

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN usermod -aG plugdev,dialout $USERNAME

RUN mkdir -p /workspaces && chown -R $USERNAME:$USERNAME /workspaces

USER $USERNAME

CMD ["/bin/bash"]