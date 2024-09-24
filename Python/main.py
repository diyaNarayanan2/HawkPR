from HawkPR import HawkPR

Alpha =0
Beta =0
EMitr = 10
Delta = 3
DaysPred=6
SimTimes=6
Crop = [100, 206]

OutputPath_pred = r"C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\output\model_pred.csv";
OutputPath_mdl = r"C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\output\mdl.mat";
InputPath_demography = r"C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\input_data\Demo_Dconfirmed.csv";
InputPath_report = r"C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\input_data\NYT_Dconfirmed.csv";
InputPath_mobility = r"C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\input_data\GoogleMobi_Dconfirmed.csv";

HawkPR(InputPath_report, InputPath_mobility, InputPath_demography, Crop, Delta, Alpha, Beta, EMitr, DaysPred, SimTimes, OutputPath_mdl, OutputPath_pred)
