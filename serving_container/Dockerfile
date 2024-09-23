FROM conda/miniconda3:latest

# copy requirements.txt

COPY . /app

# install dependencies using conda

RUN conda install scikit-learn

RUN pip install -r /app/requirements.txt

# command to run on container start, listining to connections from everywhere 

CMD ["python", "/app/app.py", "--host", "0.0.0.0", "--port", "4242"]
