FROM continuumio/miniconda3:24.9.2-0

WORKDIR /workspace

COPY environment.yml ./environment.yml
RUN conda env create -f environment.yml && conda clean -afy

COPY configs ./configs
COPY competition_artifacts ./competition_artifacts
COPY docs ./docs
COPY html_demo ./html_demo
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests
COPY README.md ./README.md

ENV PYTHONPATH=/workspace/src
ENV PATH=/opt/conda/envs/jiebang/bin:$PATH

CMD ["bash", "scripts/run_docker_smoke.sh"]
