FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN python -m pip install --no-cache-dir '.[webapi]'
# Trusted model bundles must be mounted outside the image at runtime.
# DO NOT copy artifacts/, data/, publisher test clips or any pickles here.
USER 65534:65534
EXPOSE 8000
CMD ["uvicorn","gym_pose.web_api:app","--host","0.0.0.0","--port","8000","--no-access-log","--workers","1"]
