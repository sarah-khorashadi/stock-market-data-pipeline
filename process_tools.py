import finpy_tse as fpy
import os
import pandas as pd
from pytse_client import download_client_types_records
from sklearn.preprocessing import StandardScaler
# from jdatetime import datetime, date
import jdatetime
from datetime import datetime
import matplotlib.pyplot as plt
from persiantools.jdatetime import JalaliDate
import numpy as np
import matplotlib.pylab as plt
from numpy.fft import fft, ifft
def read_stockList_file(file_name):
    try:
        with open(file_name, 'r',encoding='utf-8') as file:
            names_str = file.read()
        stocks_list = (names_str.split(','))
        print(stocks_list)
        return stocks_list
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")

iran_stock_holidays = [
    '2023-03-04',  # Nowruz (Persian New Year)
    '2023-03-11',  # Nowruz (Persian New Year)
    '2023-03-18',  # Nowruz (Persian New Year)
    '2023-04-08',  # Nowruz (Persian New Year)
    '2023-04-20',  # Nowruz (Persian New Year)
    '2023-04-21',  # Nowruz (Persian New Year)
    '2023-04-23',
    '2023-05-16',
    '2023-06-04',
    '2023-06-05',
    '2023-09-06',
    '2023-09-16',
    '2023-09-24',
    '2023-10-03',
    # Add other holidays here
]

# Convert to jdatetime date
iran_stock_holidays = [jdatetime.date.fromgregorian(date=pd.to_datetime(date).date()) for date in iran_stock_holidays]
print('iran_stock_holidays',iran_stock_holidays)


# Function to check if a date is a holiday
def is_holiday(date):
    return date in iran_stock_holidays or date.weekday() in [3, 4]  # 3 and 4 correspond to Thursday and Friday in jdatetime


def download_data(list_of_stocks, start_date, end_date, path_to_save):
    for stock in list_of_stocks:
        data = fpy.Get_Price_History(
            stock=stock,
            start_date=start_date,
            end_date=end_date,
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=True)
        data.reset_index(drop=False, inplace=True)
        print(data)
        # Create a complete date range excluding holidays and weekends
        complete_date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
        complete_date_range = [date for date in complete_date_range if not is_holiday(date)]
        print(complete_date_range)

        data = data.set_index('Date')
        data = data.reindex(complete_date_range)
        print('+++++',data)
        data.ffill(inplace=True)
        if os.path.exists(path_to_save):
            print('folder exists')
            data.to_csv(path_to_save + stock + '.csv', index_label='J-Date')
        else:
            os.makedirs(os.path.join('data', 'raw_data', 'stocks'))
            data.to_csv(path_to_save + stock + '.csv', index_label='J-Date')

def process_stock_data(path_to_save_raw,path_to_save,labeling_method):
    for filename in os.listdir(path_to_save_raw):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(path_to_save_raw, filename)
            data = pd.read_csv(file_path)
            # data = data.reset_index(drop=True)
            # remove unwanted columns
            data = data.drop(columns=['No','Market','Value','Name'])
            #create new features
            data['Final_yes'] = data['Final'].shift(1)
            # normalize our data
            data['high_rate'] = (data['High'] - data['Final_yes']) / data['Final_yes']
            data['low_rate'] = (data['Low'] - data['Final_yes']) / data['Final_yes']
            data["open_rate"] = (data["Open"] - data["Final_yes"]) / data["Final_yes"]
            data['earning_rate'] = (data['Close'] - data['Final_yes']) / data['Final_yes']
            data['Volume'] = StandardScaler().fit_transform(data[['Volume']]).flatten()
            close_prices = np.array(data["Close"])
            yf = fft(close_prices)
            yf[100:] = 0
            data['smooth-close'] = ifft(yf).real
            # create labels
            if labeling_method==0:
                data['label'] = data["earning_rate"].shift(-1).apply(lambda x: 1 if x > 0 else 0)
            elif labeling_method==1:
                data['MovingAverage'] = data['Close'].rolling(window=3, min_periods=1, center=True).mean()
                data['diff_MovingAvrage'] =  data['MovingAverage'].shift(-1)-data['MovingAverage']
                data['label'] = data['diff_MovingAvrage'].apply(lambda x: 1 if x > 0 else 0)
                data = data.drop(columns=['MovingAverage','diff_MovingAvrage'])
            elif labeling_method==2:
                # Iterate over each row in the DataFrame
                num_rows_to_sum = 5
                for index, row in data.iterrows():
                    # Calculate the sum of 'roc' for the current row and the next 20 rows
                    sum_value = data.loc[index:index+num_rows_to_sum, 'earning_rate'].sum()
                    # Replace the 'roc' value in the current row with the calculated sum
                    data.at[index, 'roc'] = sum_value
                    print(data.columns)
                    data['label'] = data['roc'].apply(lambda x: 1 if x > 0 else 0)
                # Print the modified DataFrame
                print(data)
            elif labeling_method==3:
                data['diff_smooth-close'] =  data['smooth-close'].shift(-1)-data['smooth-close']
                data['label'] = data['diff_smooth-close'].apply(lambda x: 1 if x > 0 else 0)
                data = data.drop(columns=['smooth-close','diff_smooth-close'])


        # data = data.drop(columns=['Open','High','Close','Low','Final','Final_yes'])
        data = data.dropna()
        data.to_csv(path_to_save+filename, index=False)

def save_to_csv_alltogether(merge_IndCor_datas_path_index,root_data):
    for filename in os.listdir(merge_IndCor_datas_path_index):
        if filename.endswith('.csv'):
            file_path = os.path.join(merge_IndCor_datas_path_index, filename)
            data = pd.read_csv(file_path)

            print(data.shape)
            print(data)
        if os.path.exists(root_data + 'historical_data.csv'):
            df_existing = pd.read_csv(root_data + 'historical_data.csv')
            df_combined = pd.concat([df_existing, data], ignore_index=True)
            df_combined.to_csv(root_data+ 'historical_data.csv', index=False)
        else:
            data.to_csv(root_data+ 'historical_data.csv', index=False)

# def save_to_csv_alltogether(folder_path_stock, folder_path_index,path_to_save,index_as_node):
#     for filename in os.listdir(folder_path_stock):
#         if filename.endswith('.csv'):  # Check if the file is a CSV file
#             file_path = os.path.join(folder_path_stock, filename)
#             data = pd.read_csv(file_path)
#             data = data.reset_index(drop=True)
#         if os.path.exists(path_to_save + 'historical_data.csv'):
#                 df_existing = pd.read_csv('./data/raw_data/stocks/historical_data.csv')
#                 df_combined = pd.concat([df_existing, data], ignore_index=True)
#                 df_combined.to_csv(path_to_save+ 'historical_data.csv', index=False)
#         else:
#             data.to_csv(path_to_save+ 'historical_data.csv', index=False)
#     if index_as_node:
#         for filename in os.listdir(folder_path_index):
#             if filename.endswith('.csv'):  # Check if the file is a CSV file
#                 file_path = os.path.join(folder_path_index, filename)
#                 data = pd.read_csv(file_path)
#                 data = data.reset_index(drop=True)
#             if os.path.exists(path_to_save + 'historical_data.csv'):
#                     df_existing = pd.read_csv('./data/raw_data/stocks/historical_data.csv')
#                     df_combined = pd.concat([df_existing, data], ignore_index=True)
#                     df_combined.to_csv(path_to_save+ 'historical_data.csv', index=False)
#             else:
#                 data.to_csv(path_to_save+ 'historical_data.csv', index=False)

def get_date_intersect(file_path, stocks):
    data = pd.read_csv(file_path)
    data = data.reset_index(drop=True)
    # ticker_count = data['Ticker'].unique()
    date_counts = data['Date'].value_counts()
    print(date_counts)
    common_dates = date_counts[date_counts == 93].index
    data = data[data['Date'].isin(common_dates)]
    return data

def get_overall_index(start_date, end_date):
    fpy.Get_CWI_History(
        start_date=start_date,
        end_date=end_date,
        ignore_date=False,
        just_adj_close=False,
        show_weekday=False,
        double_date=False)

def download_index(start_date,end_date,path_to_save):
        # دریافت سابقه شاخص کل
        CWI = fpy.Get_CWI_History(
            start_date= start_date,
            end_date= end_date,
            ignore_date=False,
            just_adj_close=False,
            show_weekday=False,
            double_date=False)
        CWI['Ticker'] = 'شاخص کل'
        CWI.to_csv(path_to_save+'CWI'+'.csv')
        
        # دریافت سابقه شاخص کل هم‌وزن
        EWI = fpy.Get_EWI_History(
            start_date=start_date,
            end_date=end_date,
            ignore_date=False,
            just_adj_close=False,
            show_weekday=False,
            double_date=False)
        EWI['Ticker'] = 'شاخص کل هم‌وزن'
        EWI.to_csv(path_to_save+'EWI'+'.csv')
        
        # دریافت سابقه شاخص قیمت وزنی-ارزشی
        # CWPI = fpy.Get_CWPI_History(
        #     start_date=start_date,
        #     end_date=end_date,
        #     ignore_date=False,
        #     just_adj_close=False,
        #     show_weekday=False,
        #     double_date=False)
        # CWPI.to_csv(path_to_save+'CWPI'+'.csv')

        # # دریافت سابقه شاخص قیمت هم‌وزن
        # EWPI = fpy.Get_EWPI_History(
        #     start_date=start_date,
        #     end_date=end_date,
        #     ignore_date=False,
        #     just_adj_close=False,
        #     show_weekday=False,
        #     double_date=False)
        # EWPI.to_csv(path_to_save+'EWPI'+'.csv')

        # دریافت سابقه شاخص سهام آزاد شناور
        # FFI = fpy.Get_FFI_History(
        #     start_date=start_date,
        #     end_date=end_date,
        #     ignore_date=False,
        #     just_adj_close=False,
        #     show_weekday=False,
        #     double_date=False)
        # FFI.to_csv(path_to_save+'FFI'+'.csv')

        #  دریافت سابقه شاخص بازار اول
        # MKT1I = fpy.Get_MKT1I_History(
        #     start_date=start_date,
        #     end_date=end_date,
        #     ignore_date=False,
        #     just_adj_close=False,
        #     show_weekday=False,
        #     double_date=False)
        # MKT1I.to_csv(path_to_save+'MKT1I'+'.csv')

        # دریافت سابقه شاخص بازار دوم
        MKT2I = fpy.Get_MKT2I_History(
            start_date=start_date,
            end_date=end_date,
            ignore_date=False,
            just_adj_close=False,
            show_weekday=False,
            double_date=False)
        MKT2I['Ticker'] = 'شاخص بازار دوم'
        MKT2I.to_csv(path_to_save+'MKT2I'+'.csv')

def process_index_data(folder_path, path_to_save, labeling_method):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            # data = data.reset_index(drop=True)
            data['Final_yes'] = data['Close'].shift(1)
            # normalize our data
            data['high_rate'] = (data['High'] - data['Final_yes']) / data['Final_yes']
            data['low_rate'] = (data['Low'] - data['Final_yes']) / data['Final_yes']
            data["open_rate"] = (data["Open"] - data["Final_yes"]) / data["Final_yes"]
            data['earning_rate'] = (data['Close'] - data['Final_yes']) / data['Final_yes']
            data['Volume'] = StandardScaler().fit_transform(data[['Volume']]).flatten()

            # create labels
            if labeling_method==0:
                data['label'] = data["earning_rate"].shift(-1).apply(lambda x: 1 if x > 0 else 0)
            elif labeling_method==1:
                data['MovingAverage'] = data['Close'].rolling(window=11, min_periods=1, center=True).mean()
                data['diff_MovingAvrage'] =  data['MovingAverage'].shift(-1)-data['MovingAverage']
                data['label'] = data['diff_MovingAvrage'].apply(lambda x: 1 if x > 0 else 0)
                data = data.drop(columns=['MovingAverage','diff_MovingAvrage'])

            # drop rows containing missing values
            data = data.drop(columns=['Open','High','Close','Low','Adj Close','Final_yes'])
            data = data.dropna()
            data.to_csv(path_to_save+filename,index=False)

def download_individual_corporate(stock_list, start_date,end_date):
    for stock in stock_list:
        download_client_types_records(stock, write_to_csv=True, base_path='./data/raw_data/indCor/')

def convert_to_jalali(gregorian_date):
    try:
        gregorian_datetime = datetime.strptime(gregorian_date, '%Y-%m-%d')
        jalali_date = jdatetime.datetime.fromgregorian(date=gregorian_datetime)
        return jalali_date.strftime('%Y-%m-%d')
    except ValueError:
        return None  # Handle invalid dates by returning None

def process_individual_corporate(list_of_stocks, indCor_to_save_raw, indCor_to_save_process):
    scaler = StandardScaler()
    print(list_of_stocks)
    for stock in list_of_stocks:
        file_path = os.path.join(indCor_to_save_raw, stock +'.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            # Specify the date range
            data['date'] = pd.to_datetime(data['date'])
            start_date = pd.to_datetime('2018-12-15')
            end_date = pd.to_datetime('2023-12-13')
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            data.rename(columns={'date': 'Date'}, inplace=True)
            data.drop(columns=['Unnamed: 0'], inplace=True)
            # create dates
            # normalize
            columns_to_normalize = ['individual_buy_count','corporate_buy_count','individual_sell_count','corporate_sell_count','individual_buy_vol','corporate_buy_vol','individual_sell_vol','corporate_sell_vol','individual_buy_value','corporate_buy_value','individual_sell_value','corporate_sell_value','individual_buy_mean_price','individual_sell_mean_price','corporate_buy_mean_price','corporate_sell_mean_price','individual_ownership_change']
            data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
            # Rename columns
            complete_date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
            data = data.set_index('Date')
            data = data.reindex(complete_date_range)
            data.ffill(inplace=True)
            data.bfill(inplace=True)
            data.to_csv(indCor_to_save_process+stock +'.csv', index_label='Date', index=True)

def merge_IndCor_datas(stock_list, sotck_data_path, indCor_data_path,merge_IndCor_datas_path):
    stock_path = []
    indcor_path = []
    for filename in os.listdir(indCor_data_path):
        if filename.endswith('.csv'): 
            
            stock_path = os.path.join(sotck_data_path,filename)
            indcor_path = os.path.join(indCor_data_path,filename)
            
            stock_data = pd.read_csv(stock_path)
            indCor_data = pd.read_csv(indcor_path)

            # Merge DataFrames based on 'date'
            # C:\Users\sanaz\OneDrive\Desktop\Final Project\preprocess_V1\process\client_types_data_processed\آسیا.csv
            
            merged_df = pd.merge(stock_data, indCor_data, on='Date', how='inner')
            merged_df.ffill(inplace=True)
            
            merged_df.to_csv(merge_IndCor_datas_path+filename)

def merge_IndCor_datas_index(list_of_stocks, merge_IndCor_datas_path, index_path_to_save_processed, merge_IndCor_datas_path_index):
    for stock in list_of_stocks:
        IndCor_datas_path = os.path.join(merge_IndCor_datas_path, stock + '.csv')
        IndCor_datas = pd.read_csv(IndCor_datas_path)
        IndCor_datas.reset_index(drop=True)
        for filename in os.listdir(index_path_to_save_processed):
            index_path = os.path.join(index_path_to_save_processed,filename)
            index_data = pd.read_csv(index_path)
            index_name = filename.rstrip('.csv')
            new_column_names = ['J-Date','Volume_'+index_name, 'high_rate_'+index_name, 'low_rate_'+index_name, 'open_rate_'+index_name, 'earning_rate_'+index_name, 'label_'+index_name]
            index_data = index_data.rename(columns=dict(zip(index_data.columns, new_column_names)))
            # Merge DataFrames based on 'date'
            IndCor_datas = pd.merge(IndCor_datas, index_data, on='J-Date', how='inner')
        
        IndCor_datas.drop(columns='Unnamed: 0', inplace=True)
        IndCor_datas.to_csv(merge_IndCor_datas_path_index+stock+'.csv', index=False)

    # indcors_path = [filename for filename in os.listdir(indCor_data_path) if filename.endswith('.csv')]
    # stocks_path = [filename for filename in os.listdir(sotck_data_path) if filename.endswith('.csv')]
    # print('indcors_path',indcors_path)
    # print('stocks_path',stocks_path)

    

