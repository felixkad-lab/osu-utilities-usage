import sys
import pandas as pd
import data_manipulation as dm
import helper_functions as helper

# Preprocessing data
def util_preprocessing(
        prefix, df, top_up, cleanup=False, new_columns=False,
        fix_credits=False, time_elapsed=False, ghs_used=False, ghs_used_phr=False, reorder_columns=False, save_df_tofile=False, breakup_data=False, breakup_info=None, start_end_dates=None, interpol_missing=False):    
    print("="*60)
    if prefix == 'elec':
        print("Preprocessing Electricity data\n")
    elif prefix == 'wifi':
        print("Preprocessing WiFi data\n")
    else:
        print("I don't know which utility data you want to analyze, so quitting!")
        sys.exit(1)
        
    # remove blank rows, replace NaN, create 'datetime', 'day_count',
    # 'week' columns    
    if cleanup:        
        # Procedures to perform
        set_date_range = True 
        interpol_missing = True
        rename_cols = True
        replace_nan = True
        change_dtypes = True
        reset_index = True

        # Arguments for the procedures
        columns_to_rename = {
            prefix + '_time': 'time', 
            prefix + '_measured': 'measured'
        }
        columns_to_replace_nan = {'measured': 0} #, 'time': '00:00:00'}
        start_end_dates = [
            pd.to_datetime(start_end_dates[0]).normalize(),
            pd.to_datetime(start_end_dates[1]).normalize()
        ]
        
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
        
        df = dm.cleanup_data(
            df, set_date_range, start_end_dates, interpol_missing,
            interpol_dict, rename_cols, columns_to_rename, replace_nan,
            columns_to_replace_nan, change_dtypes, columns_dtype_change,
            reset_index
        )
    
    # Create datetime, day_count, and week columns
    if new_columns:
        df = helper.create_new_columns(
            df, create_datetime=True, create_day=True, create_week=True
        )
        
    # Fix credits using top_up amount list
    if fix_credits:        
        # Create 'credit_ghs' column for wifi
        if prefix == 'wifi':
            total_ghs = sum(top_up[0])
            total_gb = sum(top_up[1])
            top_up = top_up[0]
            df['credit_ghs'] = df['credit_gb'] * total_ghs / total_gb
        
        outname = 'output/' + prefix + '_topup.tsv'
        df, latest_topup_date, latest_topup_amount, credit_max, day_max =\
            helper.fix_credit_with_topups(df, top_up, outname)
       
    # Calculate time_elapsed, ghs_used, ghs_used_per_hr
    if time_elapsed:
        df = helper.calc_time_elapsed(df)
    
    # Calculate ghs_used
    if ghs_used:
        df, ghs_used_max = helper.calc_ghs_used(df)
    
    # Calculate time_elapsed, ghs_used, ghs_used_per_hr
    if ghs_used_phr:
        df, ghs_used_phr_max = helper.calc_ghs_used_phr(df)
    
    # Breaking electricity into different parts
    if breakup_data:
        df = helper.break_data_into_parts(
            df, breakup_info['points'], breakup_info['column']
        )

    # Reorder columns
    if reorder_columns:
        if prefix == 'elec':
            new_order = [
                'date', 'measured', 'day_count', 'week', 'credit_kwh', 'credit_ghs', 'time_elapsed', 'credit_ghs_fixed', 'ac_use',
                'ghs_used', 'ghs_used_phr', 'part'
            ]
        elif prefix == 'wifi':
            new_order = [
                'date', 'measured', 'day_count', 'week', 'credit_gb', 'credit_ghs', 'time_elapsed', 'credit_ghs_fixed', 'ghs_used',
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
    if save_df_tofile:
        outname = f'output/{prefix}_data.tsv'
        print(f"Saving {prefix} dataframe to {outname}\n")
        df_formatted = df.copy()
        if prefix == 'elec':
            two_dec = [
                'credit_kwh', 'credit_ghs', 'credit_ghs_fixed', 'time_elapsed', 'ghs_used'
            ]
            #integer = ['measured', 'day_count', 'week', 'ac_use', 'part']
            integer = ['day_count', 'week']
            columns_to_save = [
                'date', 'measured', 'day_count', 'week', 'time_elapsed', 'credit_kwh', 'credit_ghs', 'credit_ghs_fixed', 'ac_use', 'ghs_used', 'ghs_used_phr', 'part'
            ]
        elif prefix == 'wifi':
            two_dec = [
                'credit_gb', 'credit_ghs', 'credit_ghs_fixed', 'time_elapsed',
                'ghs_used'
            ]
            integer = ['day_count', 'week']
            columns_to_save = [
                'date', 'measured', 'day_count', 'week', 'time_elapsed', 'credit_gb', 'credit_ghs', 'credit_ghs_fixed', 'ghs_used', 'ghs_used_phr'
            ]
        else:
            print(f"Can't save dataframe for prefix = {prefix}. Quitting")
            sys.exit(1)
                    
        df_formatted['date'] = df_formatted['date'].dt.strftime('%Y-%m-%d')
        df_formatted['ghs_used_phr'] = (
            df_formatted['ghs_used_phr']
            .apply(lambda x: dm.format_decimal(x,3))
        )
        for col in two_dec:
            df_formatted[col] = (
                df_formatted[col].apply(lambda x: dm.format_decimal(x, 2))
            )
            
        for col in integer:
            df_formatted[col] = (
                df_formatted[col].apply(lambda x: dm.format_decimal(x, 0))
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
    
    # Find Today's Credit
    today, ghs_today, today_data_exist = helper.find_today_credit(df)

    # Return Results
    return (
        df, df.iloc[-1], latest_topup_date, latest_topup_amount, day_max,
        credit_max, ghs_used_max, ghs_used_phr_max)
        