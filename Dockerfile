FROM continuumio/miniconda3:24.9.2-0

WORKDIR /workspace

COPY environment.yml ./environment.yml
RUN conda env create -f environment.yml && conda clean -afy

COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts
COPY src ./src
COPY README.md ./README.md

ENV PYTHONPATH=/workspace/src
ENV PATH=/opt/conda/envs/jiebang/bin:$PATH

CMD ["python", "scripts/check_project.py"]
