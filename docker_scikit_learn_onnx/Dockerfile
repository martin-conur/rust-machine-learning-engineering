# build stage
FROM rust:latest AS builder
WORKDIR /usr/src/app
COPY . .
RUN cargo build --release


# runtime stage

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY --from=builder /usr/src/app/target/release/docker_scikit_learn_onnx /usr/local/bin/docker_scikit_learn_onnx
COPY toy_model_generation/rf_model.onnx /usr/src/app/toy_model_generation/rf_model.onnx
EXPOSE 3000
CMD ["docker_scikit_learn_onnx"]