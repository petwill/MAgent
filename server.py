from flask import Flask, jsonify, render_template, request, Response
from keras.models import load_model
from keras import backend as K
import numpy as np

app = Flask(__name__)

@app.route('/go', methods=['GET', 'POST'])
def infer_action():
    model = load_model('fuckin_model')
    
    test = np.load('tmp.npy')
    print(test)
    res = model.predict_classes(test.reshape(1, 22))[0]
    K.clear_session()
    return Response(('{}'.format(res)).encode('utf-8'))

if __name__ == '__main__':
    app.run(host="localhost", port=12345)
