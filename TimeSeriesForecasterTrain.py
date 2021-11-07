import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import pickle

model_path = "./price_model"
input_size = 1
hidden_size = 50
num_layers = 1
num_classes = 1
seq_length = 13

def get_available_products():
    return ["Mint Sauce", "Pilau Rice", "Curry", "Chicken Tikka Masala", "Saag Aloo", "Mixed Starter", "Egg Rice", "Onion Bhaji"]

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out
if __name__ == "__main__":
    products = get_available_products()
    for prod in products:
        restaurant_training_set = pd.read_csv('archive/restaurant-1-orders.csv')
        restaurant_training_set['Order Date'] = pd.to_datetime(restaurant_training_set['Order Date'])

        training_set = restaurant_training_set[restaurant_training_set["Item Name"] == prod].groupby(pd.Grouper(key='Order Date', axis=0, freq='1D', sort=True))["Product Price"].size()
        training_set = training_set.iloc[:].values.reshape(training_set.count(), 1)

        # plt.plot(training_set, label='Restaurant Order Sum')
        # plt.show()

        def sliding_windows(data, seq_length):
            x = []
            y = []

            for i in range(len(data)-seq_length-1):
                _x = data[i:(i+seq_length)]
                _y = data[i+seq_length]
                x.append(_x)
                y.append(_y)

            return np.array(x), np.array(y)

        sc = MinMaxScaler()
        training_data = sc.fit_transform(training_set)

        x, y = sliding_windows(training_data, seq_length)

        train_size = int(len(y) * 0.67)
        test_size = len(y) - train_size

        dataX = Variable(torch.Tensor(np.array(x)))
        dataY = Variable(torch.Tensor(np.array(y)))

        trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

        testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
        testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

        num_epochs = 2000
        learning_rate = 0.01

        lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
        print(lstm)
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            outputs = lstm(trainX)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, trainY)

            loss.backward()

            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        lstm.eval()
        train_predict = lstm(dataX)

        data_predict = train_predict.data.numpy()
        dataY_plot = dataY.data.numpy()

        data_predict = sc.inverse_transform(data_predict)
        dataY_plot = sc.inverse_transform(dataY_plot)

        # plt.plot(dataY_plot[0:100])
        # plt.plot(data_predict[0:100])
        # plt.suptitle('Time-Series Prediction')
        # plt.show()

        data_predict = np.clip(data_predict, 0, 10000)

        save_computed = {
            'orig': dataY_plot,
            'predicted': data_predict,
            'date': restaurant_training_set.groupby(pd.Grouper(key='Order Date', axis=0, freq='1D', sort=True)).head()['Order Date'].iloc[:]
        }
        a_file = open(prod+".pkl", "wb")
        pickle.dump(save_computed, a_file)
        a_file.close()
        torch.save(lstm.state_dict(), model_path)

