from flask import Flask,render_template,request
import pickle
import numpy as np
import xgboost as xgb


app=Flask(__name__)

clf=pickle.load(open('model/model2.pkl','rb'))
tfv=pickle.load(open('model/tfv.pkl','rb'))
svd=pickle.load(open('model/svd.pkl','rb'))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['post'])
def predict():
    # recieve form data here
    content = request.form.get('content')

    X=np.array([content])
    #print(X)
    X_tfv=tfv.transform(X)
    X_svd=svd.transform(X_tfv)


    y_pred = clf.predict(X_svd)
    print(y_pred)

    def result(value):
       if value==0:
         return ' not bot'
       else:
         return 'bot'

    # FOR DISPLAYING THE RESULT
    return render_template('index.html', result=result(y_pred[0]))






if __name__=="__main__":
    # IF WE KEEP DEBUG=TRUE THEN THE CHANGES ARE AUTOMATICALLY REFLECTED IN THE WEBPAGE
    app.run(debug=True)