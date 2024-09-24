import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
from scipy.interpolate import RegularGridInterpolator

def covid_tr_ext_j(covid_tr, n_day_tr):
    #extends covid_tr vertically
    return np.repeat(covid_tr, n_day_tr, axis=0)

def covid_tr_ext_i(covid_tr, n_day_tr, n_cty):
    #extends covid_tr horizontally
    return np.tile(covid_tr.T, (1, n_day_tr)).T

def plot2dFunc(kernel, timeList, countyList, x_label='X-axis', y_label='Y-axis', title='Heatmap', ngrid=100):
    """
    Generates a continuous gradient heatmap from a 2D kernel with grid lines and custom labels.

    Parameters:
    kernel (2D np.array): The input 2D array representing the function over 2 dimensions.
    should be of shape n_cty x n_day
    timeList (list of str): Labels for the x-axis (time markers).
    countyList (list of str): Labels for the y-axis (county markers).
    x_label (str): Label for the X-axis.
    y_label (str): Label for the Y-axis.
    title (str): The title of the heatmap.
    ngrid (int): Resolution of the final image (ngrid x ngrid).

    Returns:
    None: Displays the heatmap.
    """
    # Get the size of the kernel
    x = np.linspace(0, 1, kernel.shape[1]) #n_days
    y = np.linspace(0, 1, kernel.shape[0]) #n_cty
    
    # Interpolation function using RegularGridInterpolator with linear interpolation
    interp_func = RegularGridInterpolator((y, x), kernel, method='cubic')
    
    # Generate new grid with higher resolution
    x_new = np.linspace(0, 1, ngrid)
    y_new = np.linspace(0, 1, ngrid)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
    
    # Interpolate the values on the new grid
    points = np.array([y_new_grid.ravel(), x_new_grid.ravel()]).T
    kernel_interp = interp_func(points).reshape((ngrid, ngrid))

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(kernel_interp, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar()
    
    # Set custom labels for x and y axes
    plt.xticks(np.linspace(0, 1, len(timeList)), timeList)
    plt.yticks(np.linspace(0, 1, len(countyList)), countyList)
    
    #grid lines
    plt.grid(axis='x', color='white', linestyle='--', linewidth=0.5)
    plt.grid(axis='y', color='red', linestyle='--', linewidth=0.5)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    
def snipDataset(InputPath_Data, DateSnip, CountySnip, rowheaders): 
    '''Code will return three arrays: 1 for the data with only values
    also sets NaN values to zero 
    one of the columnKeyList, one of the rowKeyList
    Parameters:
    InputPath_Data: sepcifies the input path to dataset
    Datesnip: specifies number of days to crop: columns
    CountySnip: specifies number of counties to crop: rows 
    rowheaders: number of types of rowheaders (eg: county name, countyID, State)
    colheaders: number of types of rowheaders (eg: dates) '''
    Data = pd.read_csv(InputPath_Data)
    if DateSnip == None:
        Data_Values = Data.iloc[:CountySnip, rowheaders:].values
        Data_Values[np.isnan(Data_Values)] = 0
        rowKeyList = Data.iloc[:CountySnip, :rowheaders]
        colKeyList = Data.columns[rowheaders:]
    else: 
        Data_Values = Data.iloc[:CountySnip, rowheaders: rowheaders+ DateSnip].values
        Data_Values[np.isnan(Data_Values)] = 0
        rowKeyList = Data.iloc[:CountySnip, :rowheaders]
        colKeyList = Data.columns[rowheaders:rowheaders+DateSnip]
    
    return Data_Values, rowKeyList, colKeyList 

def plot_covid_predictions(DaysPred, n_day_tr, covid_og, pred_cases, dates_list , compare):
    '''Function plots original cases against predicted cases 
    DaysPred is number of predicted days 
    n_day_tr is number of days used for training 
    covid_og: original dataset with case count day wise
    pred_cases: sim_out day wise summed over counties 
    dates_list: list of all dates (predicted and og)
    compare: boolean value for comparision yes/no
    '''
    # Error checking
    if covid_og.shape[1] != (n_day_tr +DaysPred):
        raise ValueError(f"covid_og should have {n_day_tr +DaysPred} days, but got {covid_og.shape[1]} days.")
    
    if len(pred_cases) != DaysPred:
        raise ValueError(f"pred_cases should have {DaysPred} days, but got {len(pred_cases)} days.")
    
    if len(dates_list) != (n_day_tr +DaysPred):
        raise ValueError(f"dates_list should have {n_day_tr +DaysPred} entries, but got {len(dates_list)} entries.")
    
    # Sum cases along axis=0 to get daily cases across all counties
    daily_cases = np.sum(covid_og, axis=0)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    # Plot the original daily cases used for training
    Date_List = [datetime.strptime(date_str[1:].replace('_', '-'), '%Y-%m-%d') for date_str in dates_list]
    plt.plot(Date_List[:n_day_tr], daily_cases[:n_day_tr], label="True Cases", color="blue")
    
    # If compare is True, highlight the last DaysPred days and plot predicted cases
    if compare:
        plt.plot(dates_list[-DaysPred:], daily_cases[-DaysPred:], label="True Cases", color="green")
        plt.plot(dates_list[-DaysPred:], pred_cases, label="Predicted Cases", color="red")
    else: 
        plt.plot(dates_list[-DaysPred:], pred_cases, label="Predicted Cases", color="red")        
    
    # Labeling the graph
    plt.xlabel("Time")
    plt.ylabel("Case Count")
    plt.title("Daily COVID-19 Cases and Model Predictions")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
