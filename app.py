from flask import Flask,render_template,url_for,request
import joblib
import numpy as np
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        model=joblib.load("svm_iris.pkl")
        sl=float(request.form["sl"])
        sw=float(request.form["sw"])
        pl=float(request.form["pl"])
        pw=float(request.form["pw"])
        arr=np.array([[sl,sw,pl,pw]])
        pred=model.predict(arr)
        return render_template("result.html",ans=pred)


