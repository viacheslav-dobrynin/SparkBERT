#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

shopt -s nullglob
found=0
for build_dir in "$root_dir"/target/*/build/faiss-sys-*/out; do
    if [[ -d "$build_dir/lib64" && ! -e "$build_dir/lib" ]]; then
        (
            cd "$build_dir"
            ln -s lib64 lib
        )
        printf 'Linked %s -> lib64\n' "$build_dir/lib"
        found=1
    fi
done

if [[ $found -eq 0 ]]; then
    echo "No faiss-sys build artifacts requiring a lib->lib64 link were found." >&2
    echo "Run 'cargo build' with the static feature first so faiss-sys generates them." >&2
    exit 1
fi
