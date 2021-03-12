#!/usr/bin/env bash
HERE=$(dirname "$0")
MLNET='mlnet'
VERSION="$1"
DOTNET_INSTALL_DIR="$HERE/.dotnet"
MLNET="$DOTNET_INSTALL_DIR/mlnet"
DOTNET="$DOTNET_INSTALL_DIR/dotnet"
SOURCE="$2"
# install mlnet if necessary
if [[ ! -x "$MLNET" ]]; then
    if [[ ! -x "$DOTNET" ]]; then
        wget https://dot.net/v1/dotnet-install.sh   
        ./dotnet-install.sh -c Current --install-dir "$DOTNET_INSTALL_DIR" -Channel 3.1
    fi

    $DOTNET --version
    $DOTNET tool install mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
else
$DOTNET tool update mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
fi

$MLNET --version
