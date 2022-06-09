import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':
    # input merchandise
    # read history data
    # extract date and merchandise
    # save data as pandas dataframe
    merchandise = input("merchandise:")
    data = pd.read_csv('history.csv', encoding= 'unicode_escape', header=[0])
    data = data[['date',merchandise]]

    # get train data to train ai model
    # use last 30 days data to predict current data
    # train data range from 2010/04/01 to 2018/12/28
    # use spread rather than price to improve accuracy
    x_train = []
    y_train = []
    for i in range(0, 2136):
        temp = []
        for j in range(0, 31):
            if j == 30:
                y_train.append(data[merchandise][i+j] - data[merchandise][i])
            else:
                temp.append(data[merchandise][i+j] - data[merchandise][i])
        x_train.append(temp)

    # get test data to predict price
    # test data range from 2019/01/02 to 2022/05/26
    x_test = []
    y_test = []
    x_base = []
    for i in range(2136, 2960):
        temp = []
        x_base.append(data[merchandise][i])
        for j in range(0, 31):
            if j == 30:
                y_test.append(data[merchandise][i+j] - data[merchandise][i])
            else:
                temp.append(data[merchandise][i+j] - data[merchandise][i])
        x_test.append(temp)
    
    # use RandomForest as model to train ai
    # use ai model to predict 
    # x_train is last 30 days spread and y_train is now spread
    model = RandomForestRegressor(n_estimators=200).fit(x_train, y_train)        
    y_predict = model.predict(x_test)

    # add spread and base price to get predict price
    for i in range(len(x_base)):
        y_predict[i] = y_predict[i] + x_base[i]

    # make plot
    date_plot = data['date'][2166:2990]
    data_plot = data[merchandise][2166:2990]
    date_plot.reset_index(inplace = True, drop = True)
    data_plot.reset_index(inplace = True, drop = True)
    plt.plot(date_plot, y_predict)
    plt.plot(date_plot, data_plot)
    plt.legend(['Predict','Origin'])
    ax = plt.subplot(111)
    plt.title("RandomForest Prediction")    
    ax.set_xlabel('date')
    ax.set_ylabel(merchandise, rotation = 0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(51))
    plt.show()
    
    # count deviation
    # the goal is to reduce this value
    error = 0
    for i in range(len(data_plot)):
        error = error + abs(data_plot[i] - y_predict[i])/data_plot[i]
    print('Average error: ' + "{:.4f}".format(error/len(data_plot)*100) + '%')

    # (optional) output ai model as pickle file
    with open('save/Random_ForestPrediction.pickle', 'wb') as f:
        pickle.dump(model, f)