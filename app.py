from flask import Flask, render_template, request, jsonify
import flask

from flask_cors import CORS

from chat import generate_answer2
from chat2 import generate_answer

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 60

CORS(app)

class MyFlask(flask.Flask):
    def get_send_file_max_age(self, name):
        if name.lower().endswith('.js'):
            return 60
        return super(MyFlask, self).get_send_file_max_age(name)

@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response

@app.get("/")
def index_get():
    return render_template("base.html")

@app.get("/test2")
def getpage():
    return render_template("gpt2.html")

@app.post("/predict")
#@app.route('/predict', methods=['POST']) 
def predict():
    text = request.form["message"]
    #text = request.get_json().get("message")
    response = generate_answer2(text)
    #message = {"answer": response}
    #return jsonify(message)
    return response

@app.post("/gpt2")
#@app.route('/predict', methods=['POST']) 
def predict_gpt2():
    text = request.form["message"]
    #text = request.get_json().get("message")
    response = generate_answer(text)
    #message = {"answer": response}
    #return jsonify(message)
    return response    

# @app.route("/get", methods=["POST"])
# def chatbot_response():
#     msg = request.form["msg"]
#     if msg.startswith('my name is'):
#         name = msg[11:]
#         ints = predict_class(msg, model)
#         res1 = getResponse(ints, intents)
#         res =res1.replace("{n}",name)
#     elif msg.startswith('hi my name is'):
#         name = msg[14:]
#         ints = predict_class(msg, model)
#         res1 = getResponse(ints, intents)
#         res =res1.replace("{n}",name)
#     else:
#         ints = predict_class(msg, model)
#         res = getResponse(ints, intents)
#     return res
if __name__ == "__main__":
    app.run(debug=True)   