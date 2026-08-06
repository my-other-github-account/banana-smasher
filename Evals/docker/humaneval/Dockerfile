FROM python:3.10-slim-bookworm@sha256:019e31cc1f52e4139d338bca44f27b71339a9da261738fd80fab9cf4595152fc

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Evals/requirements-humaneval.txt /tmp/requirements-humaneval.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-humaneval.txt

# Cache the exact dataset and derived ground truth while the build still has network and a writable root.
RUN python -c "from evalplus.evaluate import get_groundtruth, get_human_eval_plus, get_human_eval_plus_hash; problems = get_human_eval_plus(); get_groundtruth(problems, get_human_eval_plus_hash(), [])"

COPY Evals /app/Evals
ENTRYPOINT ["python", "-m", "Evals.tools.humaneval"]
