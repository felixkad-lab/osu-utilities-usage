from preprocess_data import util_preprocessing
from google_utils import json_keyfile, calendar_id
from descriptive_stats import (
    do_elec_descriptive_stats, do_wifi_descriptive_stats
)
from helper_functions import (
    make_predictions, restrict_elapsed_time, update_google_calendar
)
import linear_regression_helper as lrh
from ml_helper import predict_ac_use

def do_electricity_analysis(df_util):
    # Perform analysis for electricity data
    print("@"*80)
    print("Doing analysis for Electricity")
    print("@"*80)
    prefix='elec'

    # Electricity topup amounts in GHS
    top_up = [
         262.71, 487.57, 487.57, 487.57, 500.00, 487.57, 487.57, 487.57, 700, 
         487.57, 500, 487.57, 587.57, 187.57
    ]

    # The data will be broken into different parts based on flatmate situation
    breakup_info = {
        'points': ['2023-03-07', '2023-06-20', '2023-12-22'],
        'column': 'date'
    }
    start_end_dates = ['2023-02-16', '2024-03-12']
        
    # Preprocess data
    df_util, last_row, latest_topup_date = util_preprocessing(
        prefix, df_util, top_up, breakup_data=True, 
        breakup_info=breakup_info, start_end_dates=start_end_dates
    )

    # Copy of dataframe for for classification algorithm before any
    # filtering
    df_ml = df_util

    # Do descriptive stats for electricity data
    do_elec_descriptive_stats(
        prefix=prefix, df_util=df_util, top_up=top_up
    )

    # Restrict time-elapsed between subsequent measurements to between
    # 23 and 25 hours
    df_util = restrict_elapsed_time(
        df=df_util, min_time_elapsed=23, max_time_elapsed=25
    )

    # Do Linear Regression for all electricity data and graph
    fig_ctr = 14
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_credit_vs_day_overall.png'
    )            
    lrh.elec_linear_regression_all(df_util=df_util, figname=figname)

    # Do linear Regression based on data part (flatmate situation)
    fig_ctr = 15
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_credit_vs_day_parts.png'
    )
    m_list, b_list, scipy_linreg = lrh.elec_linear_regression_parts(
        df_util=df_util, figname=figname
    )

    # Make prediction about when credit will run out
    today_date, runout_date, buy_date, runout_day_err =\
        make_predictions(
            m_list=m_list, scipy_linreg=scipy_linreg, last_row=last_row,
            data_part=4
        )

    # Update google calendar
    pred_file = f'output/{prefix}_predictions.tsv'
    update_google_calendar(
        prefix=prefix, calendar_id=calendar_id, json_keyfile=json_keyfile, 
        pred_file=pred_file, today_date=today_date, 
        latest_topup_date=latest_topup_date, runout_date=runout_date, 
        runout_day_err=runout_day_err, buy_date=buy_date
    )
        
    # Does amount of credit used predict whether AC was used?
    # Predict whether AC was used based on Credit used
    fig_ctr = 16
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_confusion_matrix.png'
    )
    outname = 'output/elec_ac_prediction.tsv'
    predict_ac_use(data=df_ml, figname=figname, outname=outname)

def do_wifi_analysis(df_util):
    # Perform analysis for Wifi data
    print("@"*80)
    print("Doing analysis for Wifi")
    print("@"*80)
    prefix='wifi'

    # Wifi topup amounts in GHS and GB [[GHS], [GB]]
    top_up = [
        [
            350.00, 350.00, 350.00, 350.00, 350.00, 350.00, 350.00, 400.00,
            350.00
        ],
        [
            375.69, 375.69, 375.69, 375.69, 375.69, 375.69, 326.68, 373.35,
            326.68
        ]
    ]
    start_end_dates = ['2023-03-31', '2024-03-12']
        
    # Preprocess data
    df_util, last_row, latest_topup_date = util_preprocessing(
        prefix, df_util, top_up, breakup_data=False, 
        breakup_info=None, start_end_dates=start_end_dates
    )

    # Do descriptive stats for Wifi data
    do_wifi_descriptive_stats(
        prefix=prefix, df_util=df_util, top_up=top_up
    )

    # Restrict time-elapsed between subsequent measurements to between
    # 23 and 25 hours
    df_util = restrict_elapsed_time(
        df=df_util, min_time_elapsed=23, max_time_elapsed=25
    )

    # Do Linear Regression for all Wifi data and graph
    fig_ctr = 8
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_credit_vs_day_overall.png'
    )            
    m_list, b_list, scipy_linreg =\
        lrh.wifi_linear_regression_all(df_util=df_util, figname=figname)

    # Make prediction about when credit will run out
    today_date, runout_date, buy_date, runout_day_err =\
        make_predictions(
            m_list=m_list, scipy_linreg=scipy_linreg, last_row=last_row,
            data_part=1
        )

    # Update google calendar
    pred_file = f'output/{prefix}_predictions.tsv'
    update_google_calendar(
        prefix=prefix, calendar_id=calendar_id, json_keyfile=json_keyfile, 
        pred_file=pred_file, today_date=today_date, 
        latest_topup_date=latest_topup_date, runout_date=runout_date, 
        runout_day_err=runout_day_err, buy_date=buy_date
    )
    