# news_analyze_sentiment

## Download Dataset from below link 
Please download the dataset from below link 
1. model_best: [Here](https://www.dropbox.com/s/5skzsbpbk8wha7j/model-best.zip?dl=0)
2. final: [Here](https://www.dropbox.com/s/6ozggyzbwoz2cnf/final.zip?dl=0)

Extract it inside the same directory as main.py

## Downloading necessary libraries
```bash
  pip install -r requirements.txt
```

## Setting Up Coref Server
The Coref Server is part of the innovation in the classification segment. 
The article is sent through a coreference resolver to identify references that points to the same entity and aggregate their score.
<br><br>
Please download the [model.tar.gz](https://www.dropbox.com/s/b4aj5jll9tf6icr/model.tar.gz?dl=0). <br>
Place the model.tar.gz file in the same directory as the server.py.


### On Linux Machines
The coreference resolution server must be run on a different virtual environment from the main server
due to package conflicts. The server requires allennlp==2.2.0 which
can be installed on linux with.
```bash
    pip install allennlp==2.2.0
```


And the server can be ran with :
```bash
    python server.py
```

### Docker

If your machine supports a linux container, follow these steps.
<br><br>
Go into the corefServer directory and build the docker image with:
```bash
    docker image build -t image_name ./
```

This may take anywhere from 15 to 30 minutes as we need to install these packages on the docker image :
* pip
* python
* pytorch
* allennlp<br>

Then, starting the container will automatically set the server to listen to port 19000:
```bash
  docker run -p 19000:19000 --name container_name image_name 
```

###Non Linux Machines
Alternatively we can use any linux VM/ container and run:
```bash
    sudo apt-get install python3
    sudo apt-get install python3-pip
    pip install allennlp==2.2.0
```

and start the server with :
```bash
  python3 server.py
```

Note that the Coref Server needs port 19000 to be open to communicate 
with the main server.

## Running the Web Application
```bash
  streamlit run main.py
```

## Demo Video
https://user-images.githubusercontent.com/10554125/162609549-1ed75904-fcb5-423f-92c9-2227a73e137b.mp4
