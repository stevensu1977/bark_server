FROM anibali/pytorch:2.0.1-cuda11.8


RUN git clone https://github.com/suno-ai/bark && cd bark && pip install .
RUN pip install nltk && python -c "import nltk;nltk.download('punkt')"

WORKDIR /app

COPY requirements.new.txt requirements.txt

RUN pip install -r requirements.txt

COPY server/model.py .
RUN python -c "from model import load;load(True);"

COPY server/api.py api.py

EXPOSE 8000
CMD ["python", "api.py"]