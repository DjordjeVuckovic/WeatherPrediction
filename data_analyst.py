import pandas as pd

def calculate_nans(frame):
    for column_name in frame.columns:
        print(column_name + ' nan samples:', frame[column_name].isna().sum(), ',percent:'
              , frame[column_name].isna().sum()/len(frame)*100, '%')
def replace_nan_with_median(frame):
    for column_name in frame.columns:
        if frame[column_name].isna().sum() / len(frame) * 100 >0:
            filter_table[column_name].fillna(df[column_name].median(), inplace=True)

def replace_nan_with_mean(frame):
    for column_name in frame.columns:
        if frame[column_name].isna().sum() / len(frame) * 100 > 0:
          filter_table[column_name].fillna(df[column_name].mean(), inplace=True)

def one_hot_encode(column_name, data_frame):
    print('Unique values for column:',data_frame['cbwd'].unique())
    one_hot = pd.get_dummies(data_frame[column_name], dummy_na=False)
    # Drop the original 'cbwd' column and add the one-hot encoded columns
    data_frame = data_frame.drop(column_name, axis=1)
    data_frame = data_frame.join(one_hot)
    return data_frame

def calculate_min_max(frame,column):
    min_value = frame[column].min()
    max_value = df[column].max()
    print(f"Minimum value: {min_value}")
    print(f"Maximum value: {max_value}")
def unique_values(frame):
    for column_name in frame.columns:
        print(column_name)
        print(frame[column_name].unique())

df= pd.read_csv('dataset/ShenyangPM20100101_20151231.csv')
print("Rows num:", df.shape[0])
print("Cols num:", df.shape[1])
print("Data types:\n", df.dtypes, "\n")
print(df.head(10))
print("--------------------------------")
print("Missing data:\n")
missing_data=df.isnull().sum()
print(missing_data)
print(missing_data.shape)
print("--------------------------------")

#create new table without row index and PM_Caotangsi,PM_Shahepu
new_table=df.drop(['No', 'PM_Taiyuanjie', 'PM_Xiaoheyan'], axis=1)
print("New table:")
print("Shape: ",new_table.shape)
print(new_table.columns)
print(new_table.head(5))
print("--------------------------------")

#one hot encode cbwd categorical column
filter_table = one_hot_encode('cbwd',new_table)
print(filter_table.head())
print("--------------------------------")
filter_table.to_csv('dataset/categorical.csv',index=False)
calculate_nans(filter_table)
print("--------------------------------")
#dealing with nan values
replace_nan_with_median(filter_table)
calculate_nans(filter_table)
print("--------------------------------")
unique_values(filter_table)
print("--------------------------------")
calculate_min_max(filter_table,'PM_US Post')
num_rows_with_value = df['precipitation'].eq(0).sum()
print(f'Number of rows in column "precipitation" with value "0": {num_rows_with_value}')
num_rows_with_value = df['Iprec'].eq(0).sum()
print(f'Number of rows in column "Iprec" with value "0": {num_rows_with_value}')
#final_table=filter_table.drop(['Iprec', 'precipitation'], axis=1)
final_table = filter_table
final_table.to_csv('dataset/final_dataset.csv',index=False)




