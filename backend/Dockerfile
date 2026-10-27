FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
COPY pyproject.toml README.md /app/
COPY src /app/src
# Original owner RF bundles were pickled using scikit-learn 1.8.0.
# Using newer 1.9.x may silently change or break the deserialized estimator.
RUN python -m pip install --no-cache-dir '.[webapi,hfmodels]' 'scikit-learn==1.8.0'
# DO NOT COPY owner model artifacts or publisher media into this public image.
# Render Free has no SSH or persistent disk, only temporary verified downloads.
RUN groupadd --gid 10001 pose && \
    useradd --uid 10001 --gid pose --create-home --shell /usr/sbin/nologin pose && \
    mkdir -p /tmp/gym-pose-models && chown -R pose:pose /tmp/gym-pose-models
USER pose:pose
EXPOSE 8000
# No model downloads until explicit owner opt-in. A failed SHA256 pin aborts.
CMD ["sh","-c","python -m gym_pose.private_model_bootstrap && exec uvicorn gym_pose.web_api:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log --workers 1"]
