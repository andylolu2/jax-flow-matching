#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'uv generate-shell-completion fish | source' >> ~/.config/fish/config.fish
uv run echo "Hello, World!"
