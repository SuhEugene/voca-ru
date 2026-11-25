#!/bin/sh
# Install the `voca` command.
#
#     curl -fsSL https://raw.githubusercontent.com/sultson/voca/main/install.sh | sh
#
# Puts the binary in ~/.local/bin, the model in ~/.local/share/voca and, if
# Claude Code is present, the voca skill in ~/.claude/skills/voca.
# Override the first two with VOCA_BIN and VOCA_HOME.
set -eu

repo="sultson/voca"
release="https://github.com/$repo/releases/latest/download"
raw="https://raw.githubusercontent.com/$repo/main"
bin="${VOCA_BIN:-$HOME/.local/bin}"
data="${VOCA_HOME:-$HOME/.local/share/voca}"
here="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
cargo="$(command -v cargo || echo "$HOME/.cargo/bin/cargo")"

mkdir -p "$bin" "$data"

echo "Downloading the model (19 MB)..."
curl -fsSL "$release/voca-data.tar.gz" | tar -xz -C "$data"

if [ -f "$here/crates/tiny-voca/Cargo.toml" ] && [ -x "$cargo" ]; then
    echo "Building from this checkout..."
    (cd "$here" && "$cargo" build --release -p tiny-voca --bin voca)
    # Copy then rename, so a replaced binary keeps a valid signature.
    cp "$here/target/release/voca" "$bin/voca.new"
    mv -f "$bin/voca.new" "$bin/voca"
elif [ "$(uname -sm)" = "Darwin arm64" ]; then
    echo "Downloading the voca binary..."
    curl -fsSL "$release/voca-aarch64-apple-darwin.tar.gz" | tar -xz -C "$bin"
elif [ -x "$cargo" ]; then
    echo "No prebuilt binary for $(uname -sm); building with cargo..."
    "$cargo" install --quiet --locked --force --git "https://github.com/$repo" tiny-voca --bin voca --root "$(dirname "$bin")"
else
    echo "No prebuilt binary for $(uname -sm). Install Rust from https://rustup.rs and run this again." >&2
    exit 1
fi

if [ -d "$HOME/.claude" ]; then
    mkdir -p "$HOME/.claude/skills/voca"
    if [ -f "$here/skills/voca/SKILL.md" ]; then
        cp "$here/skills/voca/SKILL.md" "$HOME/.claude/skills/voca/SKILL.md"
    else
        curl -fsSL -o "$HOME/.claude/skills/voca/SKILL.md" "$raw/skills/voca/SKILL.md"
    fi
    echo "Installed the Claude Code skill."
fi

echo "Installed $bin/voca"
case ":$PATH:" in
    *":$bin:"*) echo "Try it: voca \"Привет, как дела?\"" ;;
    *) echo "Add it to your PATH, then try it:"
       echo "  export PATH=\"$bin:\$PATH\""
       echo "  voca \"Привет, как дела?\"" ;;
esac
