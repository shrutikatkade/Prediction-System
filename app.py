from flask import Flask, render_template, request,redirect,url_for
import os
import flask
import pickle
import numpy as np

from flask_bootstrap import Bootstrap
app=Flask(__name__)
Bootstrap(app)

@app.route('/send',methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('index'))
    return render_template('login.html', error=error)
    

@app.route('/index', methods = ['POST','GET'])
def index():
    return render_template('index.html')


@app.route('/result', methods = ['POST','GET'])
def result():
    result=' '
    if request.method=='POST':
        
        ans1=request.form['a']
        print(ans1)
        print(type(ans1))
        ans2=request.form['b']
        print(ans2)
        ans3=request.form['c']
        print(ans3)
        ans4=request.form['d']
        ans5=request.form['e']
        ans6=request.form['f']
        ans7=request.form['i']
        ans8=request.form['j']
        ans9=request.form['k']
        ans10=request.form['r']
        ans11=request.form['l']
        ans12=request.form['m']
        ans13=request.form['n']
        ans14=request.form['o']
        ans15=request.form['p']
        ans16=request.form['q']
        
        df=[]
        df.append(ans1)
        df.append(ans2)
        df.append(ans3)
        df.append(ans4)
        df.append(ans5)
        df.append(ans6)
        df.append(ans7)
        df.append(ans8)
        df.append(ans9)
        df.append(ans10)
        df.append(ans11)
        df.append(ans12)
        df.append(ans13)
        df.append(ans14)
        df.append(ans15)
        df.append(ans16)
        print(df)
        df_array=np.array(df).reshape(1,16)
        loaded_model = pickle.load(open('models\model.pkl','rb'))
        result =loaded_model.predict(df_array)
                
    return render_template("result.html", output=result)



if __name__=='__main__':
    app.run(debug=True)
