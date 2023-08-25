from flask import Flask, render_template, request, redirect, url_for
from backend import Character_choice
from backend import models
from backend import Visualization

__author__ = "Keimpe Dijkstra"

app = Flask(__name__)
 
@app.route('/')
def main_page():
    cc.set_choice(0)
    return render_template('index.html')

@app.route("/choice", methods=['POST'])
def choice():
    data = request.form.get('data')
    cc.set_choice(int(data))
    return data

@app.route("/questions", methods=['POST', "GET"])
def questions():
    if request.method == 'POST':
        return redirect(url_for("results"))
    return render_template("questions.html")
    

@app.route("/results", methods=["POST"])
def results():
    data = str(request.form.get('data'))
    data = data + str(cc.get_choice())

    a,b,c = m.model_factory(data)
    result = [a,b,c]
    result = round(sum(result)/len(result))
    
    return render_template("results.html",result = result, choice = cc.get_choice())
    

@app.route("/data", methods=["GET", "POST"])
def data():
    if request.method == "GET":
        Visualization.visfactory()
        return render_template("data.html")
    m.dummy_data_gen()
    Visualization.visfactory()
    return "nib"

@app.route("/alg")
def alg():
    return render_template("alg.html")


if __name__ == '__main__':
    #initialize some variables
    cc = Character_choice(0)
    m = models()

    app.run()




