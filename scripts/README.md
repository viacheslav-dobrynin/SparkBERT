# Useful scripts

[fix-faiss-libdir.sh](fix-faiss-libdir.sh)\
  Run from the repo root.\
  Creates a lib -> lib64 symlink inside `target/*/build/faiss-sys-*/out`.\
  Run after `cargo build` if linking fails with ``error: could not find native static library `faiss`, perhaps an -L flag is missing?`` or similar.\
  Confirmed useful on Fedora.
