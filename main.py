from google_utils import credentials, query
from data_import import load_utilities_data 
from analysis_types import do_electricity_analysis, do_wifi_analysis

# Main function
def main():
    # Import data and create dataframe
    df = load_utilities_data(
        credentials=credentials, query=query,
        directly=False, filename='utilities.tsv'
    )

    # Separate Electricity dataframe
    df_elec = df[
        [
            'date', 'elec_time', 'elec_measured', 'credit_ghs',
            'credit_kwh', 'ac_use'
        ]
    ]
    
    # Separate Wifi dataframe
    df_wifi = df[['date', 'wifi_time', 'wifi_measured', 'credit_gb']]

    # Perform Analysis for Electricity
    do_electricity_analysis(df_elec)  

    # Perform Analysis for Wifi
    do_wifi_analysis(df_wifi)
    print("Main script: so far so good")
    
# Execute code
if __name__ == "__main__":
    main()
