ARG N8N_VERSION=stable
FROM n8nio/n8n:${N8N_VERSION}

USER root

# Restore apk package manager for n8n Alpine image.
# The alpine version MUST match the repositories in /etc/apk/repositories of the
# n8n image (currently v3.24), otherwise installing packages replaces libraries
# that this apk binary needs and apk stops working.
COPY --from=alpine:3.24 /sbin/apk /sbin/apk
COPY --from=alpine:3.24 /lib/apk /lib/apk
COPY --from=alpine:3.24 /usr/lib/libapk* /usr/lib/

# OpenCV and numpy come from Alpine: PyPI has no musl (Alpine) wheels for
# opencv-python, so installing it with uv/pip would compile it from source.
RUN apk add --no-cache ffmpeg fontconfig font-noto python3 py3-numpy py3-opencv \
  && mkdir -p /tmp \
  && chmod 1777 /tmp

# uv: Python package manager (pinned minor version)
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    # Lets `python -m post_process.<script>` work from any directory
    PYTHONPATH=/opt

# Install the locked dependencies, except the ones provided by apk above.
# --no-deps: the export already lists every package; without it uv would try to
# satisfy scenedetect's opencv-python requirement and compile it.
WORKDIR /opt/post_process
COPY pyproject.toml uv.lock ./
RUN uv venv --system-site-packages --python /usr/bin/python3 /opt/venv \
  && uv export --frozen --no-hashes --no-emit-project --no-dev \
       --no-emit-package opencv-python --no-emit-package numpy \
       -o /tmp/requirements.txt \
  && uv pip install --no-cache --no-deps -r /tmp/requirements.txt \
  && rm /tmp/requirements.txt \
  && python -c "import cv2, scenedetect, ffmpeg; print('cv2', cv2.__version__, 'scenedetect', scenedetect.__version__)"

COPY post_process/ ./

WORKDIR /home/node
USER node
