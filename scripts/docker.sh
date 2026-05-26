#!/bin/bash

IMAGE_NAME="app-python-rust-qt"
CONTAINER_NAME="app_dev_container"

FOLDER_NAME=$(basename "$(pwd)")
WORKSPACE_DIR="/workspaces/$FOLDER_NAME"

usage() {
    echo "Uso: $0 [build | run]"
    echo "  build : Compila a imagem usando o Dockerfile do .devcontainer como contexto"
    echo "  run   : Executa o container com suporte a Qt/Open3D, SSH e privilégios idênticos ao VS Code"
    exit 1
}

if [ $# -ne 1 ]; then
    usage
fi

ACTION=$1

case "$ACTION" in
    build)
        echo "=== [BUILD] Iniciando compilação da imagem [$IMAGE_NAME] ==="
        docker build \
            --build-arg USER_UID=$(id -u) \
            --build-arg USER_GID=$(id -g) \
            -f .devcontainer/Dockerfile -t "$IMAGE_NAME" .
        ;;
        
    run)
        echo "=== [RUN] Preparando o ambiente de execução ==="
        
        xhost +local:docker > /dev/null
        
        mkdir -p "$HOME/.ssh"
        touch "$HOME/.ssh/known_hosts"
        
        echo "=== [RUN] Subindo o container gráfico em modo Privilegiado ==="
        docker run -it --rm \
            --name "$CONTAINER_NAME" \
            --net=host \
            --privileged \
            -e DISPLAY="$DISPLAY" \
            -e GIT_SSH_COMMAND="ssh -o ControlMaster=auto -o ControlPersist=60s" \
            -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
            -v "$HOME/.ssh/known_hosts:/home/vscode/.ssh/known_hosts:ro" \
            -v "$(pwd)":"$WORKSPACE_DIR" \
            -w "$WORKSPACE_DIR" \
            "$IMAGE_NAME"
            
        echo "=== [RUN] Container finalizado. Revogando permissões de tela ==="
        # Limpa as permissões de tela do xhost ao encerrar o container
        xhost -local:docker > /dev/null
        ;;
        
    *)
        usage
        ;;
esac