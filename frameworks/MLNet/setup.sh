#!/usr/bin/env bash
HERE=$(dirname "$0")
MLNET='mlnet'
VERSION=${1:-"latest"}

DOTNET_INSTALL_DIR="$HERE/lib"
MLNET="$DOTNET_INSTALL_DIR/mlnet"
DOTNET="$DOTNET_INSTALL_DIR/dotnet"
SOURCE="https://mlnetcli.blob.core.windows.net/mlnetcli/index.json"

export DOTNET_CLI_HOME="$DOTNET_INSTALL_DIR"

# if version eq latest, set Version to empty string so it will install the latest version.
if [[ "$VERSION" == "latest" ]]; then
    VERSION=""
fi

rm -rf "$DOTNET_INSTALL_DIR"
# install mlnet if necessary
if [[ ! -x "$MLNET" ]]; then
    if [[ ! -x "$DOTNET" ]]; then
        wget -P "$DOTNET_INSTALL_DIR" https://dot.net/v1/dotnet-install.sh
        chmod +x "$DOTNET_INSTALL_DIR/dotnet-install.sh"
        "$DOTNET_INSTALL_DIR/dotnet-install.sh" -c Current --install-dir "$DOTNET_INSTALL_DIR" -Channel 3.1 --verbose
    fi
    $DOTNET tool install mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
else
  $DOTNET tool update mlnet --add-source "$SOURCE" --version "$VERSION" --tool-path "$DOTNET_INSTALL_DIR"
fi

export DOTNET_ROOT="$DOTNET_INSTALL_DIR"

$MLNET --version | grep + | sed -e "s/\(.?*\)+.*/\1/" >> "${HERE}/.setup/installed"
