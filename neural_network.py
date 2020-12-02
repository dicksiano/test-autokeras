from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

class NeuralNetwork():
    def __init__(self, hyperparameters, X_train, y_train):
        self.hyperparameters = hyperparameters
        self.model = self.compile_model(X_train, X_train.shape[1], y_train.shape[1])

    def train(self, X_train, y_train, X_test, y_test):
        return self.train_model(X_train, y_train, X_test, y_test)

    def show_configuration(self):
        return '_'.join( str(i) for i in self.hyperparameters.values() )

    def compile_model(self, X, input_size, out_size):
        nb_layers  = self.hyperparameters['nb_layers']
        nb_neurons = self.hyperparameters['nb_neurons']
        activation = self.hyperparameters['activation']
        optimizer  = self.hyperparameters['optimizer']

        model = Sequential()
        model.add(Dense(nb_neurons, activation=activation, input_shape=(input_size,)))

        for layer in range(nb_layers - 1):
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dense(out_size, activation=activation))
        model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error','mean_absolute_percentage_error']) 

        return model

    def train_model(self, X_train, y_train, X_test, y_test):
        return self.model.fit(X_train, y_train, nb_epoch=10, validation_data=(X_test, y_test), shuffle=True)

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=1)