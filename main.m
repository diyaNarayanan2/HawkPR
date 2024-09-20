%scale of weibull
Alpha = 0
%shape of weibull
Beta = 0

% num of maximum iterations for EM algortihm in case convergence not reached
EMitr = 20

%additional days to be predicted by trained hawks process model
DaysPred = 6

%mobility shift parameter: ???
Delta = 3

SimTimes = 6

%to_csv function will automatically create a csv file with this path
OutputPath_pred = "C:\\Users\\dipiy\\OneDrive\\Documents\\GitHub\\CausalSTPP\\TestData"
OutputPath_mdl = "C:\\Users\\dipiy\\OneDrive\\Documents\\GitHub\\CausalSTPP\\TestData"
InputPath_demography = "C:\\Users\\dipiy\\OneDrive\\Documents\\GitHub\\CausalSTPP\\TestData\\Demography_Test.csv"
InputPath_report = "C:\\Users\\dipiy\\OneDrive\\Documents\\GitHub\\CausalSTPP\\TestData\\Report_Test2.csv"
InputPath_mobility = "C:\\Users\\dipiy\\OneDrive\\Documents\\GitHub\\CausalSTPP\\TestData\\Mobility_Test.csv"