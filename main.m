%scale of weibull
Alpha = 0;
%shape of weibull
Beta = 0;

% num of maximum iterations for EM algortihm in case convergence not reached
EMitr = 20;

%additional days to be predicted by trained hawks process model
DaysPred = 6;

%mobility shift parameter: ???
Delta = 3;

SimTimes = 6;

%to_csv function will automatically create a csv file with this path
OutputPath_pred = "C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\output\model_pred.csv";
OutputPath_mdl = "C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\output\mdl.mat";
InputPath_demography = "C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\input_data\Demo_Dconfirmed.csv";
InputPath_report = "C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\input_data\NYT_Dconfirmed.csv";
InputPath_mobility = "C:\Users\dipiy\OneDrive\Documents\GitHub\HawkPR\input_data\GoogleMobi_Dconfirmed.csv";

HawkPR(InputPath_report, InputPath_mobility, InputPath_demography, Delta, Alpha, Beta, EMitr, DaysPred, SimTimes, OutputPath_mdl, OutputPath_pred);