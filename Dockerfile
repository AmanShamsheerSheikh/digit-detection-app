# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

ARG PYTHON_VERSION=3.9.13
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.

# WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.

# Copy the source code into the container.
COPY . .
# Expose the port that the application listens on.
# Run the application.
# CMD uvicorn 'main:app' --host=127.0.0.1 --port=8000 --reload --reload-dir=/app
EXPOSE 8000
CMD uvicorn main:app --host=0.0.0.0 --reload

# bash -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app"
#