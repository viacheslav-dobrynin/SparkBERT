FROM ubuntu:latest AS builder
ENV PATH="/root/.cargo/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      intel-mkl \
      libssl-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY benches ./benches
COPY src ./src
COPY examples ./examples
RUN cargo build --release --example server

FROM ubuntu:latest
ENV SPARKBERT_INVERTED_INDEX_DIR=/data/index
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      intel-mkl \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/examples/server /app/server
EXPOSE 8000
CMD ["/app/server"]
