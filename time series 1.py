

import numpy as np
import pandas as pd
import pandas.tseries.offsets as ofst
import matplotlib
import matplotlib.pyplot as plt

# change the font size
matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)

# files we'll be working with
files=['american.csv', 'columbia.csv']

# folder with data
data_folder = ''

# read the data
american = pd.read_csv(data_folder + files[0], 
    index_col=0, parse_dates=[0], 
   header=0, names=['','american_flow'])

columbia = pd.read_csv(data_folder + files[1], 
    index_col=0, parse_dates=[0],
    header=0, names=['','columbia_flow'])

# combine the datasets
riverFlows = american.combine_first(columbia)
#riverFlow = american.combine_first(columbia)
#riverFlow.dropna(inplace=True)
riverFlows.dropna(inplace=True)

# periods aren't equal in the two datasets so find the overlap
# find the first month where the flow is missing for american
idx_american = riverFlows \
    .index[riverFlows['american_flow'].apply(np.isnan)].min()

# find the last month where the flow is missing for columbia
idx_columbia = riverFlows \
    .index[riverFlows['columbia_flow'].apply(np.isnan)].max()

# truncate the time series
riverFlows = riverFlows.truncate(
    before=idx_columbia + ofst.DateOffset(months=1),
    after=idx_american - ofst.DateOffset(months=1))

# write the truncated dataset to a file

# index is a DatetimeIndex
print('\nIndex of riverFlows')
print(riverFlows.index)

# selecting time series data
print('\ncsv_read[\'1933\':\'1934-06\']')
print(riverFlows['1933':'1934-06'])

# shifting the data
by_month = riverFlows.shift(1, freq='M')
print('\nShifting one month forward')
print(by_month.head(6))

by_year = riverFlows.shift(12, freq='M')
print('\nShifting one year forward')
print(by_year.head(6))

# averaging by quarter
quarter = riverFlows.resample('Q').mean()
print('\nAveraging by quarter')
print(quarter.head(2))

# averaging by half a year
half = riverFlows.resample('6M').mean()
print('\nAveraging by half a year')
print(half.head(2))

# averaging by year
year = riverFlows.resample('A').mean()
print('\nAveraging by year')
print(year.head(2))

# plot time series
data_folder_out = ''

# colors 
colors = ['r','y']

# monthly time series
riverFlows.plot(title='Monthly river flows', color=colors)
plt.savefig(data_folder_out + 'monthly_riverFlows.png',
    dpi=300)
plt.close()

# quarterly time series
quarter.plot(title='Quarterly river flows', color=colors)
plt.savefig(data_folder_out + 'quarterly_riverFlows.png',
    dpi=300)
plt.close()
