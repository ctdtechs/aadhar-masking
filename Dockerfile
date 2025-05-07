FROM python:3.11-slim
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
&& rm -rf /var/lib/apt/lists/*
 
RUN python -m venv /aadharmasking
 
COPY requirements.txt ./
RUN /aadharmasking/bin/pip install --no-cache-dir -r requirements.txt
 
ENV PATH="/aadharmasking/bin:$PATH"
 
COPY . .

ENV FLASK_APP=app.py
ENV FLASK_ENV=production
 
EXPOSE 5041

CMD ["python", "app.py"]