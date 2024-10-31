from flask import Flask
from flask_cors import CORS

class server:
    app = Flask(__name__)
    CORS(app)
    def __init__(self):
        pass

    @app.route("/test")
    def haloworld():
        return "Halo World"
    
    @app.route("/read")
    def send(data):
        return data
    
if __name__ == "__main__":
    server().app.run(debug=True)