import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#check normal distribution of pm
dataset= pd.read_csv('dataset/final_dataset.csv')
new_table = dataset
sns.histplot(dataset['PM_US Post'], kde=True)
plt.show()
sns.histplot(np.log1p(dataset['PM_US Post']), kde=True)
plt.show()
table_year = new_table.set_index('year')
plt.boxplot([table_year.loc['2013', 'PM_US Post'], table_year.loc['2014', 'PM_US Post'], table_year.loc['2015', 'PM_US Post']])
plt.xlabel('Godina')
plt.ylabel('Koncentracija PM2.5 [ug/m3]')
plt.grid()
plt.xticks([1,2,3], ['2013', '2014', '2015'])
plt.figure()
plt.show()
#year plots
plt.boxplot([table_year.loc['2010', 'DEWP'], table_year.loc['2011', 'DEWP'], table_year.loc['2012', 'DEWP'], table_year.loc['2013', 'DEWP'], table_year.loc['2014', 'DEWP'], table_year.loc['2015', 'DEWP']])
plt.xlabel('Godina')
plt.ylabel('Temperatura rose [°C]')
plt.grid()
plt.xticks([1,2,3,4,5,6], ['2010','2011','2012','2013', '2014', '2015'])
plt.figure()
plt.show()

plt.boxplot([table_year.loc['2010', 'TEMP'], table_year.loc['2011', 'TEMP'], table_year.loc['2012', 'TEMP'], table_year.loc['2013', 'TEMP'], table_year.loc['2014', 'TEMP'], table_year.loc['2015', 'TEMP']])
plt.xlabel('Godina')
plt.ylabel('Temperatura [°C]')
plt.grid()
plt.xticks([1,2,3,4,5,6], ['2010','2011','2012','2013', '2014', '2015'])
plt.figure()
plt.show()

plt.boxplot([table_year.loc['2010', 'HUMI'], table_year.loc['2011', 'HUMI'], table_year.loc['2012', 'HUMI'], table_year.loc['2013', 'HUMI'], table_year.loc['2014', 'HUMI'], table_year.loc['2015', 'HUMI']])
plt.xlabel('Godina')
plt.ylabel('Vlaznost vazduha [%]')
plt.grid()
plt.xticks([1,2,3,4,5,6], ['2010','2011','2012','2013', '2014', '2015'])
plt.figure()
plt.show()

plt.boxplot([table_year.loc['2010', 'PRES'], table_year.loc['2011', 'PRES'], table_year.loc['2012', 'PRES'], table_year.loc['2013', 'PRES'], table_year.loc['2014', 'PRES'], table_year.loc['2015', 'PRES']])
plt.xlabel('Godina')
plt.ylabel('Vazdusni pritisak [hPa]')
plt.grid()
plt.xticks([1,2,3,4,5,6], ['2010','2011','2012','2013', '2014', '2015'])
plt.figure()
plt.show()

plt.boxplot([table_year.loc['2010', 'Iws'], table_year.loc['2011', 'Iws'], table_year.loc['2012', 'Iws'], table_year.loc['2013', 'Iws'], table_year.loc['2014', 'Iws'], table_year.loc['2015', 'Iws']])
plt.xlabel('Godina')
plt.ylabel('Brzina vetra [m/s]')
plt.grid()
plt.xticks([1,2,3,4,5,6], ['2010','2011','2012','2013', '2014', '2015'])
plt.figure()
plt.show()
#month
#gb=NovaTabela.groupby(by=['month']).mean()
# correlation matrix
corr = dataset.corr()
highly_corr_features = corr.index[abs(corr['PM_US Post']) > 0.1]
plt.figure(figsize=(10, 10))
sns.heatmap(dataset[highly_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
print(corr['PM_US Post'].sort_values(ascending=False).head(5))
fig = plt.figure(figsize=(12, 10))
# Rooms
plt.subplot(321)
sns.scatterplot(data=dataset, x='season', y='PM_US Post')
# Area
plt.subplot(322)
sns.scatterplot(data=dataset, x='PRES', y='PM_US Post')
# YearOfBuild
plt.subplot(323)
sns.scatterplot(data=dataset, x='cv', y="PM_US Post")
plt.subplot(324)
# Location
sns.scatterplot(data=dataset, x='HUMI', y="PM_US Post")
plt.subplot(325)
sns.scatterplot(data=dataset, x='year', y="PM_US Post")
plt.subplot(326)
sns.scatterplot(data=dataset, x='day', y="PM_US Post")
plt.show()
# corelatted params
plt.scatter(dataset['season'], dataset['PM_US Post'])
plt.title("season")
plt.show()