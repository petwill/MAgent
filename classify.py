from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.models import Sequential, Model
import numpy as np
from glob import glob
from tqdm import tqdm

# stateful LSTM, ref: https://github.com/youyuge34/Cosine_Stateful_Lstm/blob/master/Stateful_Lstm_Keras.ipynb
def build_model(input_dim, n):
    model = Sequential()
    # model.add(LSTM(256, batch_input_shape=(None, 1), stateful=True))
    # model.add(LSTM(256, input_shape=(timesteps, 1)))
    model.add(Dense(1024, input_dim=input_dim,activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# def train(X_train, y_train, epochs=1):
    # for _ in range(epochs):
        # model.fit(X_train, y_train, epochs=1, verbose=2, shuffle=False)
        # model.reset_states()

# def predict(X_test, y_test=None):
    # return model.predict(X_test)
    # pass

def get_data():


    train_num = 100000
    test_num = 1000
    timesteps = 51

    X_train = np.zeros((train_num, timesteps))

    counter = 0
    for fpath in tqdm(glob('mygather_diminishing_history/*.npy')):
        data = np.load(fpath).item()

        for k, it in data.items():
            print(it)
            it = it[-50:]

            print(it)
            print(sum(it)/len(it))
            input()
            X_train[counter, :-1] = it[-50:]
            counter += 1

    print(counter)
    prev = counter
    X_train[:counter, -1] = 0

    for fpath in glob('mygather_history/*.npy'):
        data = np.load(fpath).item()

        for k, it in data.items():
            # print(sum(it)/len(it))
            # print(it)
            # input()
            X_train[counter, :-1] = it[-50:]
            counter += 1

    print(counter)
    X_train[prev:, -1] = 1

    X_train = X_train[:counter, :]
    print(X_train.shape)
    """
    from sklearn.ensemble import IsolationForest
    anom = IsolationForest()
    anom.fit(X_train)
    print(anom.predict(X_train).sum())
    input()
    """
    np.random.shuffle(X_train)
    test_sz = int(X_train.shape[0] * .2)
    X_train, y_train, X_test, y_test = X_train[:-test_sz, :-1], X_train[:-test_sz, -1], X_train[-test_sz:, :-1], X_train[-test_sz:, -1]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


if __name__=='__main__':
    # convert integers to dummy variables (i.e. one hot encoded)
    from keras.utils import np_utils
    from sklearn.utils import class_weight


    train_files = glob('./tmp/*_train.npz.npy')
    test_files = glob('./tmp/*_test.npz.npy')
    X_train = np.concatenate([np.load(f) for f in train_files], axis=0)

    test_files = [f.replace('train', 'test') for f in train_files]
    y_train = np.concatenate([np.load(f) for f in test_files], axis=0)

    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

    # print(X_train)
    # while True:
        # xx = np.random.randint(100)
        # print(xx)
        # print(list(X_train[xx,:]))
        # input()

    print(X_train.shape)
    print(y_train.shape)
    X_train = X_train[:, :22]
    y_train = np_utils.to_categorical(y_train)
    # class_weights[24] = 0
    for idx , x in enumerate(list(np.sum(y_train, axis=0))):
        print('category_{}: {}, weight:{}'.format(idx, x, class_weights[idx]))
    print(y_train)
    model = build_model(X_train.shape[1], y_train.shape[1])

    # model.fit(X_train, y_train, epochs=10, class_weight=class_weights)
    test_sz = int(X_train.shape[0]*.2)
    X_test = X_train[-test_sz:, :]
    y_test = y_train[-test_sz:, :]
    X_train = X_train[:-test_sz, :]
    y_train = y_train[:-test_sz, :]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, class_weight=dict(enumerate(class_weights)))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
    model.save('fuckin_model')
    """
    while True:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, class_weight=dict(enumerate(class_weights)))
        y_pred = model.predict_classes(X_test)
        print('accuracy', accuracy_score(y_test, y_pred))

    X_train, y_train, X_test, y_test = get_data()
    print(y_train.sum(), y_test.sum())
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)



    model = build_model(X_train.shape[1])
    model.summary()
    epochs = 30
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
    model.save('tmp')
    model = load_model('tmp')

    pred = (model.predict(X_test))
    for i in range(100):
        print(X_test[i])
        print('pred', pred[i])
        print('gt', y_test[i])
        input()
    """
