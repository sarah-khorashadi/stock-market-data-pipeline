import os
import pandas as pd
import numpy as np
import pickle
from process_tools import read_stockList_file, download_data, process_stock_data, download_individual_corporate,save_to_csv_alltogether,get_date_intersect, download_index, process_index_data,process_individual_corporate,merge_IndCor_datas,merge_IndCor_datas_index
# Get the name of the text file from user input

index_as_node = True

file_name = input("Enter the name of the text file: ")  
start_date = input("Enter start_date (e.g 2011-01-01)")
end_date = input("Enter end_date (e.g 2021-01-01)")


root_data = './data/'
stock_path_to_save_raw = './data/raw_data/stocks/'
index_path_to_save_raw = './data/raw_data/indexes/'
indCor_to_save_raw = './data/raw_data/indCor'
path_to_save_processed = './data/processed_data/stocks/test/'
index_path_to_save_processed = './data/processed_data/indexes/'
indCor_to_save_process = './data/processed_data/indCor/'
merge_IndCor_datas_path = './data/processed_data/stock_with_indcor/'
merge_IndCor_datas_path_index = './data/processed_data/stock_with_indcor_index/'

list_of_stocks = read_stockList_file(file_name)
# up down = 0
# ma = 1
# roc = 2
#fourier = 3 
labeling_method = 1

download_data(list_of_stocks, start_date, end_date, stock_path_to_save_raw)
# process_stock_data(stock_path_to_save_raw,path_to_save_processed, labeling_method) #labeling method = 0 (up-down) 1 (ma with the window of 5)

# download_index(start_date, end_date,index_path_to_save_raw)
# process_index_data(index_path_to_save_raw,index_path_to_save_processed, labeling_method)

# download_individual_corporate(list_of_stocks, start_date, end_date)
# process_individual_corporate(list_of_stocks,indCor_to_save_raw,indCor_to_save_process)

# merge_IndCor_datas(list_of_stocks, path_to_save_processed, indCor_to_save_process, merge_IndCor_datas_path)
# merge_IndCor_datas_index(list_of_stocks, merge_IndCor_datas_path, index_path_to_save_processed, merge_IndCor_datas_path_index)

# save_to_csv_alltogether(merge_IndCor_datas_path,root_data)

# # save_to_csv_alltogether(path_to_save_processed, index_path_to_save_processed,stock_path_to_save_raw, index_as_node)

# intersect_data = get_date_intersect(root_data + 'historical_data.csv',list_of_stocks)
# intersect_data.to_csv(root_data+'intersect_historical_data.csv', index=False)