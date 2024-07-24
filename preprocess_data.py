import sys
import pandas as pd
import numpy as np
import helper_functions as helper
from math import ceil

# Preprocessing data
def util_preprocessing(
        prefix, df, top_up, breakup_data=False, breakup_info=None, start_end_dates=None):    
    print("="*60)
    if prefix == 'elec':
        print("Preprocessing Electricity data\n")
    elif prefix == 'wifi':
        print("Preprocessing WiFi data\n")
    else:
        print(
            "I don't know which utility data you want to analyze, so quitting!"
        )
        sys.exit(1)
        
    # Procedures to perform
    # 1. Set date ranges to consider
    # 2. interpolate missing values
    # 3. rename columns
    # 4. replace nan values
    # 5. change data types
    # 6. reset dataframe index

    # Arguments for the procedures
    start_end_dates = [
        pd.to_datetime(start_end_dates[0]).normalize(),
        pd.to_datetime(start_end_dates[1]).normalize()
    ]   
    columns_to_rename = {
        prefix + '_time': 'time', 
            prefix + '_measured': 'measured'
    }
    columns_to_replace_nan = {'measured': 0}
    if prefix == 'elec':
        interpol_dict = {
            'time': 'pad', 'credit_ghs': 'linear', 'credit_kwh': 'linear'
        }
        columns_to_replace_nan = {'measured': 0, 'ac_use': 1}            
        columns_dtype_change = {
            'measured': 'uint8', 'credit_kwh':'float64', 'ac_use': 'uint8',
            'credit_ghs':'float64'
        }
    elif prefix == 'wifi':
        interpol_dict = {'time': 'pad', 'credit_gb': 'linear'}
        columns_to_replace_nan = {'measured': 0}            
        columns_dtype_change = {'measured': 'uint8', 'credit_gb': 'float64'}
    else:
        print(f"Cleanup procedure not set up for '{prefix}'. Quitting")
        sys.exit(1)

    # Change date to datetime format
    df = df.astype({'date':'datetime64[ns]'})

    # Clean up data 
    df = cleanup_data(
        df=df, set_date_range=True, start_end_dates=start_end_dates, 
        interpol_missing=True, interpol_dict=interpol_dict, rename_cols=True, 
        columns_to_rename=columns_to_rename, replace_nan=True,
        columns_to_replace_nan=columns_to_replace_nan, change_dtypes=True, 
        columns_dtype_change=columns_dtype_change, reset_index=True
    )
    
    # Create datetime, day_count, and week columns
    df = create_new_columns(
        df, create_datetime=True, create_day=True, create_week=True
        )
        
    # Create 'credit_ghs' column for wifi
    if prefix == 'wifi':
        total_ghs = sum(top_up[0])
        total_gb = sum(top_up[1])
        top_up = top_up[1]
        df['credit_ghs'] = df['credit_gb']
        #df['credit_ghs'] = df['credit_gb'] * total_ghs / total_gb
        
    # Normalize baseline to account for topups
    outname = f'output/{prefix}_topup.tsv'
    df, latest_topup_date = normalize_baseline(
        df, top_up, outname
    )
    
    # Calculate time_elapsed, ghs_used, ghs_used_per_hr
    df = calc_time_elapsed(df)
    
    # Calculate ghs_used
    df = calc_ghs_used(df)
    
    # Calculate time_elapsed, ghs_used, ghs_used_per_hr
    df = calc_ghs_used_phr(df)
    
    # Breaking electricity into different parts
    if breakup_data:
        df = break_data_into_parts(
            df, breakup_info['points'], breakup_info['column']
        )

    # Reorder columns
    if prefix == 'elec':
        new_order = [
            'date', 'measured', 'day_count', 'week', 'credit_kwh', 
            'credit_ghs', 'time_elapsed', 'credit_ghs_fixed', 'ac_use',
            'ghs_used', 'ghs_used_phr', 'part'
        ]
    elif prefix == 'wifi':
        new_order = [
            'date', 'measured', 'day_count', 'week', 'credit_gb', 'credit_ghs', 
            'time_elapsed', 'credit_ghs_fixed', 'ghs_used',
            'ghs_used_phr'
        ]
    else:
        print(
            f"Can't reorder columns for prefix = {prefix}, because the "
            f"options are 'elec' and 'wifi'"
        )
        sys.exit(1)       
    df = df.reindex(columns=new_order)
    
    # Save Data to file
    outname = f'output/{prefix}_data.tsv'
    print(f"Saving {prefix} dataframe to {outname}\n")
    df_formatted = df.copy()
    if prefix == 'elec':
        two_dec = [
            'credit_kwh', 'credit_ghs', 'credit_ghs_fixed', 'time_elapsed', 
            'ghs_used'
        ]
        integer = ['day_count', 'week']
        columns_to_save = [
            'date', 'measured', 'day_count', 'week', 'time_elapsed', 
            'credit_kwh', 'credit_ghs', 'credit_ghs_fixed', 'ac_use', 
            'ghs_used', 'ghs_used_phr', 'part'
        ]
    elif prefix == 'wifi':
        two_dec = [
            'credit_gb', 'credit_ghs', 'credit_ghs_fixed', 'time_elapsed',
            'ghs_used'
        ]
        integer = ['day_count', 'week']
        columns_to_save = [
            'date', 'measured', 'day_count', 'week', 'time_elapsed', 
            'credit_gb', 'credit_ghs', 'credit_ghs_fixed', 'ghs_used', 
            'ghs_used_phr'
        ]
    else:
        print(f"Can't save dataframe for prefix = {prefix}. Quitting")
        sys.exit(1)
                    
    df_formatted['date'] = df_formatted['date'].dt.strftime('%Y-%m-%d')
    df_formatted['ghs_used_phr'] = (
        df_formatted['ghs_used_phr']
        .apply(lambda x: helper.format_decimal(x,3))
    )
    for col in two_dec:
        df_formatted[col] = (
            df_formatted[col].apply(lambda x: helper.format_decimal(x, 2))
        )        
    for col in integer:
        df_formatted[col] = (
            df_formatted[col].apply(lambda x: helper.format_decimal(x, 0))
        )        
    df_formatted[columns_to_save].to_csv(
        outname, sep='\t', index=True, header=True
    )
    
    # Drop 'credit_kwh' column
    if 'credit_kwh' in df.columns.to_list():
        df = df.drop(columns=['credit_kwh'], axis=1)
    
    # Drop 'credit_gb' column
    if 'credit_gb' in df.columns.to_list():
        df = df.drop(columns=['credit_gb'], axis=1)

    return df, df.iloc[-1], latest_topup_date

# Remove blank rows, rename column, replace NaN, change datatypes,
# reset_index
def cleanup_data(df=None, set_date_range=False, start_end_dates=None,
                 interpol_missing=False, interpol_dict=None,
                 rename_cols=False, columns_to_rename=None,
                 replace_nan=False, columns_to_replace_nan=None,
                 change_dtypes=False, columns_dtype_change=None,
                 reset_index=False):
    # Restrict data range to start and end dates
    if set_date_range:
        df = df[
            (df['date'] >= start_end_dates[0]) & 
            (df['date'] <= start_end_dates[1])
        ]
        print("Finished restricting data range")
        
    # Rename some columns
    if rename_cols:
        df = df.rename(columns = columns_to_rename)
        print("Finished renaming columns")
    
    # Replace NaN using 'columns_to_replace_nan' dictionary
    if replace_nan:
        df = df.fillna(value=columns_to_replace_nan)
        print("Finished replacing NaN")
    
    # Interpolate missing values
    if interpol_missing:
        for col, method in interpol_dict.items():
            df = df.replace({col: {'': np.nan}})
            if col == 'time':
                df[col] = df[col].ffill()
            else:
                df[col] = df[col].interpolate(
                    method=method, limit_direction='forward', axis=0
                )
        print("Finished Interpolation for missing values")
        
    # Change some data types
    if change_dtypes:
        df = df.astype(columns_dtype_change)
                               
    # Reset index
    if reset_index:
        df.reset_index(inplace=True, drop=True)

    return df

# Create 'datetime', 'day_count', and 'week' columns
def create_new_columns(
        df, create_datetime=False, create_day=False, create_week=False):
    # Combine 'date' and 'time' into 'datetime' and drop 'time'
    if create_datetime:
        df['datetime'] = combine_date_time(df['date'], df['time'])
        df = df.drop(['time'], axis=1)
        
    # Create day count
    if create_day:
        df['day_count'] = (df['date'] - df.iloc[0]['date']).dt.days + 1
    
    # Create Week count
    if create_week:
        start_date = df['date'].min()
        df['week'] = df['date'].apply(lambda x: (x - start_date).days // 7 + 1)

    return df

# Combine date and time columns to form datetime
def combine_date_time(date_series, time_series):
    datetime_series = pd.to_datetime(
        date_series.astype(str) + ' ' + time_series.astype(str)
    )
    return datetime_series
   

# Normalize baseline to account for top ups
def normalize_baseline(df, top_up, save_data_to):
    print("Finding dates when more credit was added\n")

    # Calculate difference between credit in subsequent days
    df['credit_diff'] = df['credit_ghs'] - df['credit_ghs'].shift(1)

    # If credit increased, then more must have been added
    df_increase = df[df['credit_diff'] > 0] 
    row_index = []
    for i in range(df_increase.shape[0]):
        row_index.append(df_increase.index[i])
    print(f'row indices = {row_index}')   
    
    # Check to make sure that len(row_index) matches with len(top_up)
    top_up_len = len(top_up)
    row_index_len = len(row_index)
    if top_up_len != row_index_len:
        print(
            f"Found {row_index_len} points where credit increased, but was "
            f"given {top_up_len} topup amounts. "
            "The two must be equal, so quitting."
        )
        sys.exit(1)
    print("Adjusting credits to account for credit additions\n")
    
    # Make a copy of credit_ghs
    df['credit_ghs_fixed'] = df['credit_ghs']

    # Topup dates dataframe
    topup_df = df.loc[row_index, ['date']].reset_index(drop=True)

    # Create topup amounts df and add to topup_df dataframe
    amount_df = pd.DataFrame(top_up, columns=['amount'])
    topup_df['topup_amount'] = amount_df['amount']

    # Save topup dates and amounts to file
    topup_df.to_csv(save_data_to, sep='\t', index=True, header=True)

    # Grab latest top up date
    latest_topup_date = topup_df.iloc[-1]['date']
    latest_topup_date = latest_topup_date.strftime('%Y-%m-%d')
    
    # Adjust credit values to reflect credit top ups
    for i, topup in enumerate(top_up):        
        if i==0: 
            # Values before 1st top up
            df.loc[0:row_index[i]-1, 'credit_ghs_fixed'] += \
                sum(top_up[i:top_up_len])
            print(
                f"i = {i}\trow_index = {row_index[i]}\tsum to add = "
                f"{sum(top_up[i:top_up_len])}"
            )
        else: 
            # All other values
            df.loc[row_index[i-1]:row_index[i]-1, 'credit_ghs_fixed'] += \
                sum(top_up[i:top_up_len])
            print(
                f"i = {i}\trow_index = {row_index[i]}\tsum to add = "
                f"{sum(top_up[i:top_up_len])}"
            )
            print(f'i = {i}\tsum to add = {sum(top_up[i:top_up_len])}')
                
    # Change data type to float
    df['credit_ghs_fixed'] = df['credit_ghs_fixed'].astype('float64')

    # Remove 'credit_diff' columns from df
    df = df.drop(['credit_diff'], axis=1)
    return df, latest_topup_date

# Calculate time elapsed      
def calc_time_elapsed(df):
    df['time_elapsed'] = df['datetime'] - df['datetime'].shift(1)
    df = df.drop(['datetime'], axis=1)
    
    # Convert elapsed time to hours (shift up one)
    df['time_elapsed'] = (
        df['time_elapsed']
        .apply(lambda x: x.total_seconds() / 3600)
        .shift(-1)
    )
    df['time_elapsed'] = df['time_elapsed'].astype('float16')
        
    return df

# Calculate credit used       
def calc_ghs_used(df):        
    df['ghs_used'] = df['credit_ghs_fixed'] - df['credit_ghs_fixed'].shift(-1)

    # maximum ghs_used_max
    ghs_used_max = df['ghs_used'].max()
    ghs_used_max = ceil(ghs_used_max / 10) * 10
        
    return df    

# Calculate credit used per hour       
def calc_ghs_used_phr(df):
    # Calculate usage_per_hour
    df['ghs_used_phr'] = df['ghs_used'] / df['time_elapsed']
    df['ghs_used_phr'] = df['ghs_used_phr'].astype('float64')
    
    # maximum ghs_used_max
    ghs_used_phr_max = df['ghs_used_phr'].max()
    ghs_used_phr_max = ceil(ghs_used_phr_max)

    return df

# Break electricity into different parts
def break_data_into_parts(df, break_points, col):
    print("Breaking electricity data into different parts as follows:")
    print("\tPart 1: Flatmate1 with general AC use")
    print("\tPart 2: Flatmate1 w/out general AC use")
    print("\tPart 3: Flatmate2 with AC use")
    print("\tPart 4: Living alone after Flatmate4 leaves \n")

    # Break_points index
    num_break_points = len(break_points)
    bp = [None] * num_break_points; 
    break_point = [None] * num_break_points
    for i in range(num_break_points):
        break_point[i] = pd.to_datetime(break_points[i]).normalize()
        bp[i] = df[df[col] == break_point[i]].index[0]
 
    # Create 'part' column in dataframe
    df['part'] = 1
    for i in range(num_break_points):
        ## last part
        if i == num_break_points - 1:
            df.loc[bp[-1]:, 'part'] = num_break_points + 1
        else:
            df.loc[bp[i]: bp[i + 1], 'part'] = i + 2            
    df['part'] = df['part'].astype('category')
                  
    return df 