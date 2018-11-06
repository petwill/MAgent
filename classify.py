from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.models import Sequential, Model
import numpy as np
from glob import glob
from tqdm import tqdm

# stateful LSTM, ref: https://github.com/youyuge34/Cosine_Stateful_Lstm/blob/master/Stateful_Lstm_Keras.ipynb
def build_model():
    model = Sequential()
    # model.add(LSTM(256, batch_input_shape=(None, 1), stateful=True))
    model.add(LSTM(1000, input_shape=(1000,1)))
    #model.add(Dense(1024, input_dim=input_dim,activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# def train(X_train, y_train, epochs=1):
    # for _ in range(epochs):
        # model.fit(X_train, y_train, epochs=1, verbose=2, shuffle=False)
        # model.reset_states()

# def predict(X_test, y_test=None):
    # return model.predict(X_test)
    # pass

def get_data():

    num = 10
    from glob import glob
    from sklearn.model_selection import train_test_split
    files = glob('./history/*.npy')
    print(len(files))
    X_train = []
    for f in files[:num]:
        X_train.append(np.load(f)[1:])
    files = glob('./dhistory/*.npy')
    print(len(files))
    for f in files[:num]:
        X_train.append(np.load(f)[1:])
    X_train = np.stack(X_train)
    print(X_train.shape)
    y_train = np.array([1]*num + [0]*num)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train.reshape(-1,X_train.shape[1],1), X_test.reshape(-1,X_test.shape[1],1), y_train, y_test


if __name__=='__main__':

    model = build_model()
    model.summary()
    X_train ,X_test, y_train, y_test = get_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, class_weight=dict(enumerate(class_weights)))
    print('fitting ...')
    model.fit(X_train, y_train, batch_size=32, verbose=1, validation_data=(X_test, y_test), epochs=20)
    model.save('fuckin_model')
