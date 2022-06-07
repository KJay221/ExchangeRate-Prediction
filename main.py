import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':
    data = pd.read_csv('history.csv', encoding= 'unicode_escape', header=[0])
    # data = data.drop('NZD/USD', 1)
    data = data[['date','USD/NTD']]
    x_train = []
    y_train = []
    for i in range(0, 2136):
        temp = []
        for j in range(0, 31):
            if j == 30:
                y_train.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
            else:
                temp.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
        x_train.append(temp)

    x_test = []
    y_test = []
    x_base = []
    for i in range(2136, 2960):
        temp = []
        x_base.append(data['USD/NTD'][i])
        for j in range(0, 31):
            if j == 30:
                y_test.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
            else:
                temp.append(data['USD/NTD'][i+j] - data['USD/NTD'][i])
        x_test.append(temp)
    
    model = RandomForestRegressor(n_estimators=200).fit(x_train, y_train)        
    y_predict = model.predict(x_test)

    for i in range(len(x_base)):
        y_predict[i] = y_predict[i] + x_base[i]

    date_plot = data['date'][2166:2990]
    data_plot = data['USD/NTD'][2166:2990]
    date_plot.reset_index(inplace = True, drop = True)
    data_plot.reset_index(inplace = True, drop = True)
    plt.plot(date_plot, y_predict)
    plt.plot(date_plot, data_plot)
    plt.legend(['Predict','Origin'])
    ax = plt.subplot(111)
    plt.title("RandomForest Prediction")    
    ax.set_xlabel('date')
    ax.set_ylabel('USD/NTD', rotation = 0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(51))
    plt.show()
    
    error = 0
    for i in range(len(data_plot)):
        error += abs(data_plot[i] - y_predict[i])
    print(error)

    with open('save/Random_ForestPrediction.pickle', 'wb') as f:
        pickle.dump(model, f)