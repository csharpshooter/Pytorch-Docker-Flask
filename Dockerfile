FROM python:3.8-slim-buster

MAINTAINER Abhijit Mali

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]

CMD ["app.py", "sample_upload_img.jpg"]