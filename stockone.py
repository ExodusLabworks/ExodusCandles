
def get_stock_data(ticker, end_date, num_trading_days):
    try:
        # Convert end date to datetime
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Estimate buffer to cover enough *trading* days
        buffer_days = int(num_trading_days * 1.5)
        start_dt = end_dt - timedelta(days=buffer_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        print(f"Requesting data for {ticker} from {start_date} to {end_date}")

        # Download data
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        if stock_data.empty:
            print("No data found. Please check the ticker or date range.")
            return None

        # Debug: Print out the first few rows of the stock data
        # print("Downloaded data (first 5 rows):")
        # print(stock_data.head())

        # Flatten MultiIndex columns if needed
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [' '.join(col).strip() for col in stock_data.columns.values]
            #print("Flattened columns:")
            #print(stock_data.columns)  # To check the new column names

        # Now extract the necessary columns with ticker prefix
        opens = stock_data[f"Open {ticker}"].astype(float).tolist()
        closes = stock_data[f"Close {ticker}"].astype(float).tolist()
        highs = stock_data[f"High {ticker}"].astype(float).tolist()
        lows = stock_data[f"Low {ticker}"].astype(float).tolist()
        volumes = stock_data[f"Volume {ticker}"].astype(int).tolist()
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()

        return dates, opens, closes, highs, lows, volumes

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def bullOrBear(result,d,ticker):
    start_array = result[2]
   # print(len(start_array))
    total_array_size = int(len(start_array))
    #print(total_array_size)
    start_value = start_array[0]
    
    end_value = start_array[total_array_size-1]
    

    if start_value > end_value:
        print(f"\n{ticker} for {d} days is experiencing a BEAR market (-):\n")
    elif end_value > start_value:
        print(f"\n{ticker} for {d} days is experiencing a BULL market (+):\n")
    else:
        print(f"\n{ticker} for {d} days is experiencing volatilty and/or limited change:\n")
        
def RSI_determination(result,d):
    close_values = result[2]

    d2 = d-14
    close_gains = 0
    close_loses =0
    for i in range(d2,d):
        if close_values[i] > close_values[i-1]:
            close_gains = close_gains + close_values[i]
        else:
            close_loses = close_loses + close_values[i]
    
    RS_VALUE = close_gains/close_loses
    RSI = 100 - (100/(1+RS_VALUE))

    return RSI

def prediction(result,d,ticker,end_date):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    medians=[]
    fibs = []
    ratios = [] #for gains and losses strings 
    alphas = [] #alphas are wick sizes corresponding to opens
    betas = [] #betas are wick sizes corresponding to closes
    sigma = [] #comparison to determine if wick is indicator 
    phi = [] #comparison to determine if wick is indicator
    accuracy = 0

    for i in range(len(highs)):
        medians.append(abs((opens[i]+closes[i])/2))## gives us a median check point for quick comparitors
        if(opens[i]>=closes[i]):
            alphas.append(highs[i])
            betas.append(lows[i]) ##These append or add to the back to ensure proper sized arrays or storage
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if(opens[i]>=closes[i]):
            sigma.append(alphas[i]-opens[i])
            phi.append(closes[i]-betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i]-alphas[i]) ## This part is determining the difference between wicks regardless of the closing days. 
            #Sigmas are upper wicks, and phi's are lower wicks despite type of day this makes it easier for size comparison.
            #Indicator wicks have to have a sufficiently large sigma or phi relative to the other
            #For instance if PHI is 1.5 and sigma is 0.002, thats a large negative wick indicating and up shift in the next few days 


    for i in range(len(opens)):
        if(i<3):
            fibs.append(closes[i])
            
        else:
            fibs.append((closes[i]+closes[i-1]+closes[i-2])/3)
    
    for i in range(len(fibs)):
        if(i==0):
            ratios.append("0") ##0's indicate no movement
            continue
        else:
            if(fibs[i]>fibs[i-1]):
                ratios.append("+")
            elif(fibs[i]<fibs[i-1]):
                ratios.append("-")
            else:
                ratios.append("0")
    
    ###CHECKING for ratio trend shifting.
    for i in range(len(sigma)-1):
        if(sigma[i]>0.2 and phi[i]<0.2):
            ##checking result,
            if(ratios[i+1] != "-"):
                ratios[i+1] = '-'
                #ratios[i] = '-'
        elif(sigma[i]<0.2 and phi[i]>0.2):
            if(ratios[i+1] != "+"):
                ratios[i+1] = '+'
                #ratios[i] = '+'
    
    ##Comparison time for accuracy
    for i in range(len(ratios)-1):
        if(ratios[i] == "+" and (medians[i+1] > medians[i])):
            accuracy = accuracy+1
        elif(ratios[i] == "-" and (medians[i+1]< medians[i])):
            accuracy = accuracy+1
    
    prediction_accuracy = (accuracy/(len(ratios)))*100


    # if (prediction_accuracy <= 50):
    #     if(ratios[d] == "+"):
    #         prediction_accuracy = 100 - prediction_accuracy
    #         print(f"{ticker} will close LOWER tomorrow than today with Inverse Fibonacci Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    #     else:
    #         prediction_accuracy = 100 - prediction_accuracy
    #         print(f"{ticker} will close HIGHER tomorrow than today with Inverse Fibonacci Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    # else:
 
    if(prediction_accuracy >= 50):
        if ratios[d-1] == "+":
            print(f"{ticker} will close likely HIGHER tomorrow than {end_date} with Fibonacci Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
        else:
            print(f"{ticker} will close likely LOWER tomorrow than {end_date} with Fibonacci Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    else:
        if ratios[d-1] == "+":
            print(f"{ticker} will close likely LOWER tomorrow than {end_date}::: INVERSE USED since::: Fibonacci Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
        else:
            print(f"{ticker} will close likely HIGHER tomorrow than {end_date}::: INVERSE USED since::: Fibonacci Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    
    

        
def SMA_(result,d):
    closes = result[2]
    SMA =0

    for i in range(d):
        SMA = closes[i]+SMA ## summation basically
    
    SMA = SMA/d ##SMA is simple moving average, the EMA is estimated and counted for weight with most recent sales 
    return SMA
   

def EMA_SMA(result, d):

    closes = result[2]
    SMA =0
    for i in range(d):
        SMA = closes[i]+SMA ## summation basically
    
    SMA = SMA/d ##SMA is simple moving average, the EMA is estimated and counted for weight with most recent sales 
   # print(f"\nSimple Moving Average: {SMA:.2f} for {ticker}, over {d} days")

    ## Our weight will be placed on the most recent 10 day window
    ## this is important for a multiplyer
    ema_multipl = 2/(d-1)

    ##Formula for EMA is  EMA = Close price in this case result[2] or closes[d] x ema multipl + EMA(from yesterday) x 1-ema multi
    for i in range(1,d):
        EMAt = (closes[i]*ema_multipl) + (closes[i-1]*(1-ema_multipl))
    
    EMA = (closes[d-1]*ema_multipl) + (EMAt*(1-ema_multipl))

    #print(f"Exponential Moving Average: {EMA:.2f} for {ticker}, over {d} days\n\n")
    return EMA



def trend_Analytic(result,d,ticker):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    medians=[]
    fibs = []
    ratios = [] #for gains and losses strings 
    alphas = [] #alphas are wick sizes corresponding to opens
    betas = [] #betas are wick sizes corresponding to closes
    sigma = [] #comparison to determine if wick is indicator 
    phi = [] #comparison to determine if wick is indicator
    trend = []
    accuracy = 0

    for i in range(len(highs)):
        medians.append(abs((opens[i]+closes[i])/2))## gives us a median check point for quick comparitors
        if(opens[i]>=closes[i]):
            alphas.append(highs[i])
            betas.append(lows[i]) ##These append or add to the back to ensure proper sized arrays or storage
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if(opens[i]>=closes[i]):
            sigma.append(alphas[i]-opens[i])
            phi.append(closes[i]-betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i]-alphas[i]) ## This part is determining the difference between wicks regardless of the closing days. 
            #Sigmas are upper wicks, and phi's are lower wicks despite type of day this makes it easier for size comparison.
            #Indicator wicks have to have a sufficiently large sigma or phi relative to the other
            #For instance if PHI is 1.5 and sigma is 0.002, thats a large negative wick indicating and up shift in the next few days 


    for i in range(len(opens)):
        if(i<3):
            fibs.append(closes[i])
            
        else:
            fibs.append((closes[i]+closes[i-1]+closes[i-2])/3)
    
    for i in range(len(fibs)):
        if(i==0):
            ratios.append("0") ##0's indicate no movement
            continue
        else:
            if(fibs[i]>fibs[i-1]):
                ratios.append("+")
            elif(fibs[i]<fibs[i-1]):
                ratios.append("-")
            else:
                ratios.append("0")
    
    ###CHECKING for ratio trend shifting.
    for i in range(len(sigma)-1):
        if(sigma[i]>0.2 and phi[i]<0.2):
            ##checking result,
            if(ratios[i+1] != "-"):
                ratios[i+1] = '-'
                #ratios[i] = '-'
        elif(sigma[i]<0.2 and phi[i]>0.2):
            if(ratios[i+1] != "+"):
                ratios[i+1] = '+'
                #ratios[i] = '+'
    
    ##Comparison time for accuracy
    for i in range(len(ratios)-1):
        if(ratios[i] == "+" and (medians[i+1] > medians[i])):
            accuracy = accuracy+1
        elif(ratios[i] == "-" and (medians[i+1]< medians[i])):
            accuracy = accuracy+1
    
    prediction_accuracy = (accuracy/(len(ratios)))*100

    ##Comparison for trend ID
    for i in range(len(sigma)-3):
        if(phi[i]>0.2 and sigma[i] < 0.2):
            if((closes[i+1] > closes[i] and closes[i+2]>(closes[i] or closes[i+1])) or (closes[i+2] > (closes[i] or closes[i+1]) and (closes[i+3] > (closes[i+1] or closes[i+2])))):
                #print("Trend Postive change")
                trend.append("+")
        elif(phi[i]<0.2 and sigma[i] > 0.2):
            if((closes[i+1] < closes[i] and closes[i+2]<(closes[i] or closes[i+1])) or (closes[i+2] < (closes[i] or closes[i+1]) and (closes[i+3] < (closes[i+1] or closes[i+2])))):
                #print("Trend Negative change")
                trend.append("-")
        else:
            trend.append("0")
    if trend[len(trend)-1] == "+":
        print(f"Trend indicated for {ticker}, is likely to shift bullish over the next few days: ")
    elif trend[len(trend)-1] == "-":
        print(f"Trend indicated for {ticker}, is likely to shift bearish over the next few days: ")
    
            
def opens_closes(result,d):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    medians=[]
    fibs = []
    ratios = [] #for gains and losses strings 
    alphas = [] #alphas are wick sizes corresponding to opens
    betas = [] #betas are wick sizes corresponding to closes
    sigma = [] #comparison to determine if wick is indicator 
    phi = [] #comparison to determine if wick is indicator
    trend = []
    accuracy = 0

    for i in range(len(highs)):
        medians.append(abs((opens[i]+closes[i])/2))## gives us a median check point for quick comparitors
        if(opens[i]>=closes[i]):
            alphas.append(highs[i])
            betas.append(lows[i]) ##These append or add to the back to ensure proper sized arrays or storage
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if(opens[i]>=closes[i]):
            sigma.append(alphas[i]-opens[i])
            phi.append(closes[i]-betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i]-alphas[i]) ## This part is determining the difference between wicks regardless of the closing days. 
            #Sigmas are upper wicks, and phi's are lower wicks despite type of day this makes it easier for size comparison.
            #Indicator wicks have to have a sufficiently large sigma or phi relative to the other
            #For instance if PHI is 1.5 and sigma is 0.002, thats a large negative wick indicating and up shift in the next few days 


    for i in range(len(opens)):
        if(i<3):
            fibs.append(closes[i])
            
        else:
            fibs.append((closes[i]+closes[i-1]+closes[i-2])/3)
    
    for i in range(len(fibs)):
        if(i==0):
            ratios.append("0") ##0's indicate no movement
            continue
        else:
            if(fibs[i]>fibs[i-1]):
                ratios.append("+")
            elif(fibs[i]<fibs[i-1]):
                ratios.append("-")
            else:
                ratios.append("0")
    
    ###CHECKING for ratio trend shifting.
    for i in range(len(sigma)-1):
        if(sigma[i]>0.2 and phi[i]<0.2):
            ##checking result,
            if(ratios[i+1] != "-"):
                ratios[i+1] = '-'
                #ratios[i] = '-'
        elif(sigma[i]<0.2 and phi[i]>0.2):
            if(ratios[i+1] != "+"):
                ratios[i+1] = '+'
                #ratios[i] = '+'
    
    ##Comparison time for accuracy
    for i in range(len(ratios)-1):
        if(ratios[i] == "+" and (closes[i+1] > opens[i+1])):
            accuracy = accuracy+1
        elif(ratios[i] == "-" and (closes[i+1]< opens[i+1])):
            accuracy = accuracy+1
    
    prediction_accuracy = (accuracy/(len(ratios)))*100



def main():
    ##BASIC MENU OPERATIONS
    print("\nWelcome to Exodus Analytics Data Analyzer...")
    print("Please choose from the following menu.")
    print("Current Version: v1.1.6\n")

    ticker = input("Enter NYSE ticker symbol (e.g., AAPL): ").upper()
    end_date = input("Enter end date (YYYY-MM-DD): ")

    print("\nEnter [1] 30d prior ")
    print("Enter [2] 60d prior ")
    print("Enter [3] 90d prior ")
    print("Enter [4] 120d prior ")
    print("Enter [5] 180d prior ")
    userMenuChoice = int(input("Enter your choice: "))


    ##ERROR HANDLING INSURES VALID CODE IS SELECTED
    while len(end_date) != 10:
        end_date = input("ERROR: DATE MUST BE YYYY-MM-DD format: ")

    while userMenuChoice > 5 or userMenuChoice < 1:
        userMenuChoice = int(input("Invalid selection. Enter a number from 1 to 5: "))

    d = {1: 30, 2: 60, 3: 90, 4: 120, 5: 180}[userMenuChoice]

    result = get_stock_data(ticker, end_date, d)

    ## FINDS RESULTS FROM GETTING STOCK DATA AND PARSES IT INTO "ARRAYS" REALLY THEY ARE DATASETS
    if result:
        dates, opens, closes, highs, lows, volumes = result
        displayVars = input(f"Do you want to display stock data for {ticker} over {d} days?[Y/N]:")
        if displayVars == 'Y' or displayVars == 'y':
            print(f"\n{len(dates)} trading days of data for {ticker} ending before {end_date}:\n")
            for i in range(len(dates)):
                print(f"{dates[i]} | Open: {opens[i]:.2f} | High: {highs[i]:.2f} | Low: {lows[i]:.2f} | Close: {closes[i]:.2f} | Volume: {volumes[i]}")
        bullOrBear(result,d,ticker)
        RSI =  RSI_determination(result,d) 
        SMA = float(SMA_(result,d))
        EMA = float(EMA_SMA(result,d))
        print(f"{ticker} for {d} days prior to {end_date} has an RSI: {RSI:.2f}")
        if RSI >=60:
            print("RSI Values greater than 70 are generally considered OVERBOUGHT")
        elif RSI <= 40:
            print("RSI Values lower than 30 are generally considered OVERSOLD")
        else:
            print("RSI Values within normal buy/sell ratio")
        print(f"\nClose price for {end_date}: {closes[d-1]:.2f}")
        print(f"Exponential Moving Average: {EMA:.2f} for {ticker}, over {d} days")
        print(f"Simple Moving Average: {SMA:.2f} for {ticker}, over {d} days\n")
        prediction(result,d,ticker,end_date)
        prediction_EMA(result, d ,ticker,end_date)
        prediction_REMA(result, d, ticker, RSI,end_date)
        trend_Analytic(result,d,ticker)
        opens_closes(result,d)
        # finalPrediction(RSI,end_date,result,d,ticker)
    else:
        print("Failed to retrieve stock data.")

    ##DETERMINES BULL OR BEAR, RSI, AND ALGORITHMIC TRENDS AND OTHER TRENDS

def prediction_EMA (result,d,ticker, end_date):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    medians=[]
    fibs = []
    ratios = [] #for gains and losses strings 
    alphas = [] #alphas are wick sizes corresponding to opens
    betas = [] #betas are wick sizes corresponding to closes
    sigma = [] #comparison to determine if wick is indicator 
    phi = [] #comparison to determine if wick is indicator
    accuracy = 0
    indicators = 0
 
    for i in range(len(highs)):
        medians.append(abs((opens[i]+closes[i])/2))## gives us a median check point for quick comparitors
        if(opens[i]>=closes[i]):
            alphas.append(highs[i])
            betas.append(lows[i]) ##These append or add to the back to ensure proper sized arrays or storage
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if(opens[i]>=closes[i]):
            sigma.append(alphas[i]-opens[i])
            phi.append(closes[i]-betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i]-alphas[i]) ## This part is determining the difference between wicks regardless of the closing days. 
            #Sigmas are upper wicks, and phi's are lower wicks despite type of day this makes it easier for size comparison.
            #Indicator wicks have to have a sufficiently large sigma or phi relative to the other
            #For instance if PHI is 1.5 and sigma is 0.002, thats a large negative wick indicating and up shift in the next few days 


    for i in range(len(opens)):
        if(i<3):
            fibs.append(closes[i])
            
        else:
            fibs.append(EMA_SMA(result,i))
    
    for i in range(len(fibs)):
        if(i==0):
            ratios.append("0") ##0's indicate no movement
            continue
        else:
            if(fibs[i]>fibs[i-1]):
                ratios.append("+")
            elif(fibs[i]<fibs[i-1]):
                ratios.append("-")
            else:
                ratios.append("0")
    
    ###CHECKING for ratio trend shifting.
    for i in range(len(sigma)-1):
        if(sigma[i]>0.15 and phi[i]<0.1):
            ##checking result,
            indicators = indicators+1
            if(ratios[i+1] != "-"):
                ratios[i+1] = '-'
                print("indicator")
                #ratios[i] = '-'
        elif(sigma[i]<0.1 and phi[i]>0.15):
            indicators = indicators+1
            if(ratios[i+1] != "+"):
                ratios[i+1] = '+'
                print("indicator")
                #ratios[i] = '+'
       # else: ratios[i] = "0"
    
    ##Comparison time for accuracy
    for i in range(len(ratios)-1):
        if(ratios[i] == "+" and (closes[i+1] > closes[i+1])):
            accuracy = accuracy+1
        elif(ratios[i] == "-" and (closes[i+1]< closes[i+1])):
            accuracy = accuracy+1
    
    prediction_accuracy = (accuracy/(len(ratios)))*100

    # if (prediction_accuracy <= 50):
    #     if(ratios[d] == "+"):
    #         prediction_accuracy = 100 - prediction_accuracy
    #         print(f"{ticker} will close LOWER tomorrow than today with Inverse EMA Algorthim: {prediction_accuracy:.2f}% accuracy over past {d} days")
    #     else:
    #         prediction_accuracy = 100 - prediction_accuracy
    #         print(f"{ticker} will close HIGHER tomorrow than today with Inverse EMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
  
    if(prediction_accuracy >= 50):
        if ratios[d-1] == "+":
            print(f"{ticker} will close likely HIGHER tomorrow than {end_date} with EMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
        else:
            print(f"{ticker} will close likely LOWER tomorrow than {end_date} with EMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    else:
        if ratios[d-1] == "+":
            print(f"{ticker} will close likely LOWER tomorrow than {end_date}::: INVERSE USED since::: EMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
        else:
            print(f"{ticker} will close likely HIGHER tomorrow than {end_date}::: INVERSE USED since::: EMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    
def prediction_REMA (result,d,ticker,RSI, end_date):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    medians=[]
    fibs = []
    day_RSI = []
    ratios = [] #for gains and losses strings 
    alphas = [] #alphas are wick sizes corresponding to opens
    betas = [] #betas are wick sizes corresponding to closes
    sigma = [] #comparison to determine if wick is indicator 
    phi = [] #comparison to determine if wick is indicator
    accuracy = 0
    indicators = 0
 
    for i in range(len(highs)):
        medians.append(abs((opens[i]+closes[i])/2))## gives us a median check point for quick comparitors
        if(opens[i]>=closes[i]):
            alphas.append(highs[i])
            betas.append(lows[i]) ##These append or add to the back to ensure proper sized arrays or storage
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if(opens[i]>=closes[i]):
            sigma.append(alphas[i]-opens[i])
            phi.append(closes[i]-betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i]-alphas[i]) ## This part is determining the difference between wicks regardless of the closing days. 
            #Sigmas are upper wicks, and phi's are lower wicks despite type of day this makes it easier for size comparison.
            #Indicator wicks have to have a sufficiently large sigma or phi relative to the other
            #For instance if PHI is 1.5 and sigma is 0.002, thats a large negative wick indicating and up shift in the next few days 


    for i in range(len(opens)):
        if(i<3):
            fibs.append(closes[i])
            
        else:
            fibs.append(EMA_SMA(result,i))
    
    for i in range(len(fibs)):
        if(i==0):
            ratios.append("0") ##0's indicate no movement
            continue
        else:
            if(fibs[i]>fibs[i-1]):
                ratios.append("+")
                indicators+=1
            elif(fibs[i]<fibs[i-1]):
                ratios.append("-")
                indicators+=1
            else:
                ratios.append("0")
    
    ###CHECKING for ratio trend shifting.
    for i in range(len(sigma)-1):
        if(sigma[i]>0.25 and phi[i]<0.15 and RSI>=70):
            ##checking result,
            if(ratios[i+1] != "-"):
                ratios[i+1] = '-'
                #ratios[i] = '-'
        elif(sigma[i]<0.15 and phi[i]>0.25 and RSI <=30):
            if(ratios[i+1] != "+"):
                ratios[i+1] = '+'
                #ratios[i] = '+'
       # else: ratios[i] = "0"
    for i in range(14,len(opens)):
        
        d2 = i
        d1 = i - 14 
        cGains = 0
        cLoses =0

        for l in range(d1,d2):
            if closes[l] > closes[l-1]:
                cGains= cGains + closes[l]
            else:
                cLoses = cLoses + closes[l]
        RS_VALUE = cGains/cLoses
        day_RSI.append(100-(100/(1+RS_VALUE)))
    for i in range(len(day_RSI)):
        if day_RSI[i] <= 30:
            ratios[i] = "+"
        elif day_RSI[i] >=70:
            ratios[i] = "-"
    
    ##Comparison time for accuracy
    for i in range(len(ratios)-1):
        if(ratios[i] == "+" and (medians[i+1] > medians[i])):
            accuracy = accuracy+1
        elif(ratios[i] == "-" and (medians[i+1]< medians[i])):
            accuracy = accuracy+1
    
    prediction_accuracy = (accuracy/indicators)*100
    if(prediction_accuracy >= 50):
        if ratios[d-1] == "+":
            print(f"{ticker} will close likely HIGHER tomorrow than {end_date} with REMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
        else:
            print(f"{ticker} will close likely LOWER tomorrow than {end_date} with REMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
    else:
        if ratios[d-1] == "+":
            print(f"{ticker} will close likely LOWER tomorrow than {end_date}::: INVERSE USED since::: REMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")
        else:
            print(f"{ticker} will close likely HIGHER tomorrow than {end_date}::: INVERSE USED since::: REMA Algorithm: {prediction_accuracy:.2f}% accuracy over past {d} days")

def finalPrediction(RSI,end_date,result,d,ticker):
    highs = result[3]
    lows = result[4]
    closes = result[2]
    opens = result[1]
    test_closes = []
    day_RSI = []
    medians=[]
    fibs = []
    Emas = []
    alpha_closes=[]
    ratios = [] #for gains and losses strings 
    alphas = [] #alphas are wick sizes corresponding to opens
    betas = [] #betas are wick sizes corresponding to closes
    sigma = [] #comparison to determine if wick is indicator 
    phi = [] #comparison to determine if wick is indicator
    accuracy = 0
 
    for i in range(len(highs)):
        medians.append(abs((opens[i]+closes[i])/2))## gives us a median check point for quick comparitors
        if(opens[i]>=closes[i]):
            alphas.append(highs[i])
            betas.append(lows[i]) ##These append or add to the back to ensure proper sized arrays or storage
        else:
            alphas.append(lows[i])
            betas.append(highs[i])
    
    for i in range(len(highs)):
        if(opens[i]>=closes[i]):
            sigma.append(alphas[i]-opens[i])
            phi.append(closes[i]-betas[i])
        else:
            sigma.append(betas[i] - closes[i])
            phi.append(opens[i]-alphas[i]) ## This part is determining the difference between wicks regardless of the closing days. 
            #Sigmas are upper wicks, and phi's are lower wicks despite type of day this makes it easier for size comparison.
            #Indicator wicks have to have a sufficiently large sigma or phi relative to the other
            #For instance if PHI is 1.5 and sigma is 0.002, thats a large negative wick indicating and up shift in the next few days 


    for i in range(len(opens)): 
        if(i<3):
            Emas.append(closes[i])
            
        else:
            Emas.append(EMA_SMA(result,i))


    for i in range(len(Emas)):#(alpha(EMA-close_t))
        if(i<3):
            alpha_closes.append(Emas[i])
        else:
            alpha_closes.append(Emas[i]-closes[i]) ## Will modify each of these for relative strength next


    for i in range(len(opens)): #(gamma(close_t-1  - closes_t))
            if (i == 0):
                fibs.append(closes[i])
            else:

                fibs.append(closes[i-1]-closes[i]) ## gamma variable
    
    for i in range(14,len(opens)):
        
        d2 = i
        d1 = i - 14 
        cGains = 0
        cLoses =0

        for l in range(d1,d2):
            if closes[l] > closes[l-1]:
                cGains= cGains + closes[l]
            else:
                cLoses = cLoses + closes[l]
        RS_VALUE = cGains/cLoses
        day_RSI.append(100-(100/(1+RS_VALUE)))


    prediction_Length = len(day_RSI)

    for i in range(prediction_Length,len(closes)):
        test_closes.append(closes[i] + (0.05*Emas[i]) + (0.05*(day_RSI[i-len(day_RSI)]-50))+ (0.05*fibs[i]))
   



if __name__ == '__main__':
    import pandas as pd  # Make sure this is available
    import yfinance as yf
    from datetime import datetime, timedelta
    main()
