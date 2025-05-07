FROM python:3.11-slim

WORKDIR /app

RUN python -m venv /aadharmasking

COPY requirements.txt ./
RUN /aadharmasking/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/aadharmasking/bin:$PATH"

COPY . .

ENV FLASK_APP=run.py
ENV FLASK_ENV=production

EXPOSE 5041

CMD ["python", "app.py"]