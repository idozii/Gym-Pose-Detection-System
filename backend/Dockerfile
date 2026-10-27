FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN python -m pip install --no-cache-dir '.[webapi]'
# Trusted model bundles must be mounted outside the image at runtime.
# DO NOT copy artifacts/, data/, publisher test clips or any pickles here.
# Render shell/SCP needs a real non-root login home with a private .ssh dir.
# The model mount lives OUTSIDE the home directory and never enters the image.
RUN groupadd --gid 10001 pose && \
    useradd --uid 10001 --gid pose --create-home --shell /bin/bash pose && \
    mkdir -p /home/pose/.ssh /private-models && \
    chmod 700 /home/pose/.ssh && \
    chown -R pose:pose /home/pose /private-models
USER pose:pose
EXPOSE 8000
# Render supplies PORT; local Docker/Compose continue to default to 8000.
CMD ["sh","-c","exec uvicorn gym_pose.web_api:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log --workers 1"]
