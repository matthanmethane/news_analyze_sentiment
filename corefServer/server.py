import socket,pickle,sys
from allennlp.predictors import Predictor




print("Loading model")
pred = Predictor.from_path('/model.tar.gz')




class corefResolver():
    def __init__(self,port,model):
        self.port = port
        self.model = model

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = "0.0.0.0"

        sock.bind((host, port))

        self.socket = sock

    def run(self):
        self.socket.listen(5)
        print("Listening on port ",self.port)
        while 1:
            try:
                client,addr = self.socket.accept()
                print("Connection from",addr)

                data = client.recv(4096)
                data = pickle.loads(data)
                print("rec",data)

                res = self.model.predict(data)
                print("pred",res)

                client.send(pickle.dumps(res['clusters']))
            except:
                pass




server = corefResolver(19000,pred)
server.run()

