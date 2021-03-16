#!/usr/bin/env bash
HERE=$(dirname "$0")
MLNET='mlnet'
VERSION="$1"
DOTNET_INSTALL_DIR="$HERE/lib"
MLNET="$DOTNET_INSTALL_DIR/mlnet"
DOTNET="$DOTNET_INSTALL_DIR/dotnet"
SOURCE="/mnt/c/Users/xiaoyuz/source/repos/machinelearning-tools/artifacts/packages/Release/Shipping/"
# install mlnet if necessary
if [[ ! -x "$MLNET" ]]; then
    if [[ ! -x "$DOTNET" ]]; then
        wget -P "$DOTNET_INSTALL_DIR" https://dot.net/v1/dotnet-install.sh   
        "$DOTNET_INSTALL_DIR/dotnet-install.sh" -c Current --install-dir "$DOTNET_INSTALL_DIR" -Channel 3.1
    fi

    $DOTNET --version
    $DOTNET tool install mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
else
$DOTNET tool update mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
fi

export DOTNET_ROOT="$DOTNET_INSTALL_DIR"
$MLNET --version
