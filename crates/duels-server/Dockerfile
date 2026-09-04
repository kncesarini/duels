# Build context is the repo root (see docker-compose.yml), because
# duels-server is a workspace member and needs its sibling crates.
FROM rust:1-slim-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY data ./data
RUN cargo build --release -p duels-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/duels-server /usr/local/bin/duels-server
ENV DUELS_SERVER_ADDR=0.0.0.0:8080
EXPOSE 8080
CMD ["duels-server"]
