#import os
import sys
import math
import pandas as pd
from googleapiclient.errors import HttpError
from stats_stuff import do_t_test, do_anova_test
from google_calendar_helper import (
    initialize_gcal, add_event_allday, delete_event
)

# Format number to n decimal places
def format_decimal(x, n):
    if pd.notnull(x):
        return "{:.{}f}".format(x, n)
    return x

# Combine date and time columns to form datetime
def combine_date_time(date_series, time_series):
    datetime_series = pd.to_datetime(
        date_series.astype(str) + ' ' + time_series.astype(str)
    )
    return datetime_series
   
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
        df = df[(df['date'] >= start_end_dates[0]) & (df['date'] <= start_end_dates[1])]
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
                df[col] = df[col].interpolate(method=method, limit_direction='forward', axis=0)
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

    # Grab latest date
    #latest_topup_amount = topup_df.iloc[-1]['topup_amount']
    latest_topup_date = topup_df.iloc[-1]['date']
    latest_topup_date = latest_topup_date.strftime('%Y-%m-%d')
    
    # Adjust credit values to reflect credit top ups
    for i, topup in enumerate(top_up):        
        if i==0: 
            # Values before 1st top up
            df.loc[0:row_index[i]-1, 'credit_ghs_fixed'] += \
                sum(top_up[0:top_up_len])
        else: 
            # All other values
            df.loc[row_index[i-1]:row_index[i]-1, 'credit_ghs_fixed'] += \
                sum(top_up[i:top_up_len])
                
    # Change data type to float
    df['credit_ghs_fixed'] = df['credit_ghs_fixed'].astype('float64')

    # Remove 'credit_diff' columns from df
    df = df.drop(['credit_diff'], axis=1)
    return df, latest_topup_date

# Calculate time elapsed      
def calc_time_elapsed(df):
    df['time_elapsed'] = df['datetime'] - df['datetime'].shift(1)
    df = df.drop(['datetime'], axis=1)
    
    ## Convert elapsed time to hours (shift up one)
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
    ghs_used_max = math.ceil(ghs_used_max / 10) * 10
        
    return df    

# Calculate credit used per hour       
def calc_ghs_used_phr(df):
    # Calculate usage_per_hour
    df['ghs_used_phr'] = df['ghs_used'] / df['time_elapsed']
    df['ghs_used_phr'] = df['ghs_used_phr'].astype('float64')
    
    # maximum ghs_used_max
    ghs_used_phr_max = df['ghs_used_phr'].max()
    ghs_used_phr_max = math.ceil(ghs_used_phr_max)

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

# Find Today's Credit
def find_today_credit(df):
    # Find today's date
    today = pd.to_datetime('today').normalize()    
    try:
        ghs_today = df.loc[df['date'] == today, 'credit_ghs_fixed'].iat[0]
        today_data_exist = True
    except IndexError:
        ghs_today = df.iloc[-1]['credit_ghs_fixed']
        today = df.iloc[-1]['date']
        today_data_exist = False

    return today, ghs_today, today_data_exist

# Use only measured values
def only_measured(prefix, df=None):
    print("="*60)
    print("Using only Measured Data\n")

    df = df[df['measured'] == 1]

    return df
    
# Retrict time_elapsed
def restrict_elapsed_time(
        df=None, min_time_elapsed=None, max_time_elapsed=None):
    print("="*60)        
    print(
        f"Restricting Elapsed time to between {min_time_elapsed} and "
        f"{max_time_elapsed}\n"
    )
    df = (
        df[
            (df['time_elapsed'] >= min_time_elapsed) 
            & (df['time_elapsed'] <= max_time_elapsed)
        ]
    )

    return df

# Calculate time elapsed percentages for pie chart
def calc_time_elapsed_pie(df=None, bin_edges=None, outname=None):
    print("Calculating elapsed time percentages for pie chart")
    # Calculate the frequencies and percentages
    df['time_elapsed'] = df['time_elapsed'].astype('float32')
    elapsed_len = len(df) - 1
    elapsed_freq = df['time_elapsed'].value_counts(bins=bin_edges, sort=False)
    elapsed_perc = 100 * elapsed_freq / elapsed_len
    perc_sum = elapsed_perc.sum()
    
    # Create labels based on bin_edges
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        label = str(bin_edges[i]) + " - " + str(bin_edges[i+1])
        bin_labels.append(label)

    # Create a dataframe
    data = {'labels': bin_labels, 'perc': elapsed_perc}
    df_elapsed = pd.DataFrame(data)
    df_elapsed.reset_index(inplace=True)
    df_elapsed = df_elapsed.drop(['index'], axis=1)

    df_elapsed.to_csv(outname, sep='\t', index=True, header=True)
    print(f"Sum of percentages = {perc_sum}\n")
    
    return df_elapsed

# Calculate usage by week
def calc_weekly_usage(df=None, outname=None):
    print("="*60)
    print("Performing Weekly analysis of credit used\n")
    week_max = df['week'].max()
    
    # Create dataframe with only the first day of each week
    weekly = df.groupby('week').first()
    weekly.reset_index(inplace=True)
    
    # Grab relevant columns
    cols = ['measured', 'date', 'day_count', 'week', 'credit_ghs_fixed']
    weekly = weekly[cols]
    
    # Calculate ghs_used
    weekly['ghs_used'] = weekly['credit_ghs_fixed'] - weekly['credit_ghs_fixed'].shift(-1)
    
    # Save data
    weekly.to_csv(outname, sep='\t', index=True, header=True)

    return weekly

# Calculate usage by month and graph
def calc_monthly_usage(prefix, df=None, top_up=None, outname=None):        
    print("="*60)
    print("Performing Monthly Analysis of credit used\n")
    # Filter dataframe for only first of month
    first_of_month = df[df['date'].dt.day == 1]
    first_of_month = first_of_month.copy()
    first_of_month['next_month'] = (first_of_month['date']
                                    + pd.offsets.MonthBegin(1))
    
    monthly = pd.merge(
        first_of_month, df, left_on='next_month', right_on='date', suffixes=('_1st', '_subseq')
    )
    monthly['ghs_used'] = (
        monthly['credit_ghs_fixed_1st'] 
        - monthly['credit_ghs_fixed_subseq']
    )
    monthly = (
        monthly[
            [
                'date_1st', 'date_subseq', 'credit_ghs_fixed_1st', 'credit_ghs_fixed_subseq', 'ghs_used'
            ]
        ]
    )
    monthly = monthly.rename(columns={'date_1st': 'date'})        
    
    # Amount used in February
    if prefix == 'elec':
        top_up_len = len(top_up)
        original_credit = 1329.54 ## This is an estimate
        march_first_credit = monthly.iloc[0]['credit_ghs_fixed_1st']
        original_credit_adj = original_credit + sum(top_up[0:top_up_len])
        #print(f"Adjusted original credit = {original_credit_adj}\n")
        first_row = pd.DataFrame(
            {
                'date': ['2023-02-01'],
                'date_subseq': ['2023-03-01'],
                'credit_ghs_fixed_1st': [original_credit_adj],
                'credit_ghs_fixed_subseq': [march_first_credit],
                'ghs_used': [original_credit_adj - march_first_credit]
            }
        )
        monthly = pd.concat([monthly, first_row], ignore_index=True)
        monthly['date'] = pd.to_datetime(monthly['date'])
    
    # Sort by date column
    monthly = monthly.sort_values(
        by=['date']
    ).reset_index(drop=True)
    
    # Add current month to dataframe
    today = pd.to_datetime('today').normalize()
    last_month_date = monthly.iloc[-1]['date']
    last_month_month = last_month_date.month
    today_day = today.day
    if today_day != 1:
        # Get today's credit
        today, ghs_today, today_data_exist = find_today_credit(df)
        
        # Add one more row to the monthly dataframe
        credit_ghs_fixed_1st = monthly.iloc[-1]['credit_ghs_fixed_subseq']
        ghs_used = credit_ghs_fixed_1st - ghs_today
        
        new_row = pd.DataFrame(
            {
                'date': [monthly.iloc[-1]['date_subseq']],
                'date_subseq': [today.strftime("%Y-%m-%d")],
                'credit_ghs_fixed_1st': [credit_ghs_fixed_1st],
                'credit_ghs_fixed_subseq': [ghs_today],
                'ghs_used': [ghs_used]
            }
        )
        monthly = pd.concat([monthly, new_row], ignore_index=True)
        
    # Save data to file
    monthly.to_csv(outname, sep='\t', index=True, header=True)

    return monthly

# T-Test for ghs_used for AC and no AC to find if difference 
# is significant                   
def ac_t_test(prefix, df=None, outname=None):
    print("="*60)
    # Data Frame for ghs_used with AC
    tmp = df.query("ac_use == 1")
    df_AC = tmp[['ghs_used']].copy()
    df_AC = df_AC.dropna(how='all', ignore_index=True)
    
    # Data Frame ghs_used with No AC
    tmp = df.query("ac_use == 0")
    df_noAC = tmp[['ghs_used']].copy()
    df_noAC = df_noAC.dropna(how='all', ignore_index=True)

    # Perform T-Test
    alpha = 0.05
    t_stat, p_value = do_t_test(
        data1=df_AC, data2=df_noAC, data1_name='Credit Used with AC',
        data2_name='Credit Used without AC', equal_var=False, alpha=alpha, outname=outname
    )
    print(f"t_stat = {t_stat}\np_value = {p_value}\n")
    
# Anova-Test for ghs_used by part                                     
def ac_anova_test(prefix, df=None, outname=None):
    print("="*60)
    # Data Frame for ghs_used by parts
    num_parts = df['part'].nunique()
    df_parts = [None] * num_parts

    for i in range(num_parts):
        tmp = df[df['part'] == i + 1]
        df_parts[i] = tmp[['ghs_used']]
        df_parts[i] = df_parts[i].dropna(how='all', ignore_index=True)
    
    # Perform Anova-Test
    data_names = [f"Part {i+1}" for i in range(num_parts)]
    f_stat, p_value = do_anova_test(
        data=df_parts, data_names=data_names, outname=outname
    )
    
    print(f"f_stat = {f_stat}\np_value = {p_value}\n")

# Make predictions                                                    
def make_predictions(
        latest_topup_date=None, m_list=None, b_list=None, scipy_linreg=None, 
        last_row=None, data_part=None):
    print("="*60)
    print(
        f"Now making Predictions for when credit will run out. Using part "
        f"{data_part} of the data.\n"
    )

    # Get slope
    slope = m_list[data_part - 1]
    #intercept = b_list[data_part - 1]

    # Get slope and intercept error
    slope_err = scipy_linreg[data_part - 1].stderr
    intercept_err = scipy_linreg[data_part - 1].intercept_stderr

    # Get Credit today
    today = last_row['date']
    ghs_today = last_row['credit_ghs_fixed']
    
    # Calculate date when credit will run out
    days_left, runout_day_err = calc_x_intercept(
        slope=slope, intercept=ghs_today, slope_err=slope_err,
        intercept_err=intercept_err
    )
    days_left = math.floor(days_left)
    runout_day_err = math.ceil(runout_day_err)
    runout_date = today + pd.to_timedelta(days_left, unit='d')
    runout_date = runout_date.strftime("%Y-%m-%d")
    
    # Date to buy more credit
    if days_left > 5:
        buy_date = today + pd.to_timedelta(days_left - 5, unit='d')
    else:
        buy_date = today
        
    buy_date = buy_date.strftime("%Y-%m-%d")
    today_date = today.strftime("%Y-%m-%d")
        
    ## Print information 
    print(f"Today's date is {today_date}")
    print(f"Today's credit is {ghs_today.round(2)} GHS\n")
    print(f"Credit will run out in {days_left} days on {runout_date}\n")
    print(f"This date has an error of +/- {runout_day_err} days")

    return (
        today_date, ghs_today, days_left, runout_date, buy_date, runout_day_err
    )

# Calculate x intercept and it's error
def calc_x_intercept(
        slope=None, intercept=None, slope_err=None, intercept_err=None):
    x_intercept = -intercept / slope
    x_intercept_err = (
        x_intercept * (
            (intercept_err / intercept) ** 2 + (slope_err / slope) ** 2
        ) ** 0.5
    )

    return x_intercept, x_intercept_err

# Update calendar with new predictions  
def update_google_calendar(
        prefix=None, calendar_id=None, json_keyfile=None, pred_file=None, today_date=None, latest_topup_date=True, runout_date=None, runout_day_err=None, buy_date=None):
    # Load Prediction file
    data = pd.read_table(pred_file)
    df_pred = pd.DataFrame(data)
        
    # Get event information for previous prediction
    last_date, last_runout_date, last_topup_date, last_eventID_pred, \
        last_eventID_buy, last_pred_error = df_pred.iloc[-1]

    utility = {'elec': 'Electricity', 'wifi': 'WiFi'}    
    # Initialize Calendar
    service = initialize_gcal(json_keyfile)
    
    # Delete last prediction from calendar and add new event
    try:
        delete_event(calendar_id, last_eventID_pred, service)
    except HttpError as e:
        if e.resp.status == 404:
            print(
                f"{utility[prefix]} runs out event with ID {last_eventID_pred}"
                " does not exist\n"
            )
        elif e.resp.status == 410:
            print(
                f"{utility[prefix]} runs out event with ID {last_eventID_pred}"
                " had already been deleted\n"
            )
        else:
            print(f"Error: {e}\n")
            
    # Delete last buy events from calendar and add new event
    try:
        delete_event(calendar_id, last_eventID_buy, service)
    except HttpError as e:
        if e.resp.status == 404:
            print(
                f"Buy {utility[prefix]} event with ID {last_eventID_buy} does"
                " not exist\n"
            )
        elif e.resp.status == 410:
            print(
                f"Buy {utility[prefix]} event with ID {last_eventID_buy} had"
                " already been deleted\n"
            )
        else:
            print(f"Error: {e}\n")
            
    # Add Electricity runs out event to calendar
    summary_text = f'{utility[prefix]} runs out (' + today_date + ' prediction)'
    eventID_pred, link_pred = add_event_allday(
        calendar_id, summary_text, runout_date, runout_date, 'GMT', service
    )
    
    # Add Buy Electricity event to calendar
    summary_text = f'Buy {utility[prefix]} (' + today_date + ' prediction)'
    eventID_buy, link_buy = add_event_allday(
        calendar_id, summary_text, buy_date, buy_date, 'GMT', service
    )
    
    ## Write calendar information to file
    with open(pred_file, "a") as file:
        #file.write("today_date\trunout_date\teventID_pred\teventID_buy\n")
        file.write(
            f"{today_date}\t{runout_date}\t{latest_topup_date}\t{eventID_pred}"
            f"\t{eventID_buy}\t{runout_day_err}\n"
        )

