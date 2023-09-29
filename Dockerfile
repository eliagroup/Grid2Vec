FROM python:3.11.4-bullseye

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"

# Install Poetry
RUN apt-get -qq update && apt-get -qq -y install curl\
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false \
    && apt-get -qq -y remove curl \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Set the working directory \
WORKDIR /app

# Copy dependencies
COPY poetry.lock pyproject.toml ./

# Install dependencies
RUN poetry install
