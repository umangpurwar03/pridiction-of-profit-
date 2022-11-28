from flask import Flask, render_template, request
import pickle
import pandas as pd
import statsmodels

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/home/')
def predict():
    scal = request.args.get('scal')
    sfl = request.args.get('sfl')
    snw = request.args.get('snw')
    rd = request.args.get('rd')
    with open('finalmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    a = {"scal": [int(scal)],
         "sfl": [int(sfl)],
         "snw": [int(snw)],
         "RD": [float(rd)]}
    test2 = pd.DataFrame(a)
    print(test2)
    resp = model.predict(test2)
    print(resp[0])
    return render_template('home.html', resp=resp[0])

if __name__ == '__main__':
    app.run()