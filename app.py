from flask import Flask, render_template,request
import model as m

app = Flask(__name__)

data_email = ""
msg = "" 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    if request.method == "POST":
        data_email = request.form["email_text"]
        class_type = m.model(data_email)
        if class_type == ['spam'] :
            msg = """<h4 style="color: red">Spam email be aware !!!!</h4>"""
        else : 
            msg = """<h4 style="color: green">email looks Good<<->> </h4>"""
    return render_template("index.html" ,msg= msg, data_email = data_email )

if __name__ == "__main__":
    app.run(debug=True)    