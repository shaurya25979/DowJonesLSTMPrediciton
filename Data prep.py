# Preparing and visualizing OHLC, HLC and CLose




for EachStock, ticker in zip(StockData,tickers):
    np.random.seed(7)
    obs = np.arange(1, len(EachStock) + 1, 1)
    OHLC_avg = EachStock.mean(axis = 1)
    HLC_avg = EachStock[['High', 'Low', 'Close']].mean(axis = 1)
    close_val= EachStock[['Close']]
    plt.figure(figsize=(15,5))
    subplot(2,1,1)
    plt.plot(obs, OHLC_avg, label = 'OHLC avg')
    plt.legend(loc = 'upper right')
    plt.title(ticker)
    
    subplot(2,1,2)
    plt.plot(obs, HLC_avg,'r', label = 'HLC avg')
    plt.plot(obs, close_val, 'g', label = 'Closing price')
    plt.legend(loc = 'upper right')
    plt.xlabel('Number of days')
    #print(HLC_avg)
    print('\n\n\n')
    plt.show()

    #color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))