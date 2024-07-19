import os
import sys
import pandas as pd
import gspread
#import psycopg2 as ps
from google.oauth2 import service_account

# Get data from file
def get_from_file(filename):
    # Get file extension
    ext = os.path.splitext(filename)[1]
    ext = ext.upper()

    # Import data depending on extension
    if ext == '.TSV':
        data = pd.read_table(filename)
        return data
    elif ext == '.CSV':
        data = pd.read_csv(filename)
        return data
    else:
        print(f"I don't know how to open the {ext} extension. Quitting")
        sys.exit(1)
    
# Get data from Google Sheet
def get_from_gsheet(credentials, query, filename):
    # Check to make sure Credential keys are present
    credential_keys = ['json_keyfile', 'scope']
    for key in credential_keys:
        if not credentials.get(key):
            print(
                f"I'm missing {key} in the credentials dictionary. "
                f"Quitting.\n"
            )
            sys.exit(1)

    json_keyfile = credentials['json_keyfile']
    scope = credentials['scope']
    creds = service_account.Credentials.from_service_account_file(
        json_keyfile, scopes=scope
    )

    # Get worksheet information from query
    query_keys = ['spreadsheet_id', 'worksheet_name', 'range_name']
    for key in query_keys:
        if not query.get(key):
            print(f"I'm missing {key} in the query dictionary. Quitting.\n")
            sys.exit(1)

    spreadsheet_id = query['spreadsheet_id']
    worksheet_name = query['worksheet_name']
    range_name = query['range_name']

    # Authorize and open the Google Sheets document
    client = gspread.authorize(creds)    
    spreadsheet = client.open_by_key(spreadsheet_id)
    worksheet = spreadsheet.worksheet(worksheet_name)

    # Get the data from the specified name
    data = worksheet.get(range_name)

    return data

# Load Data depending on source: file, google sheet, SQL database
def load_data(source, filename, credentials, query):    
    if source == 'file':
        # Load data directly from a file
        print("Loading data from file\n")
        data = get_from_file(filename)
    elif source == 'gsheet':
        # Loading directly from google sheet
        print("Loading data directly from google sheet\n")
        credentials['scope'] = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        data = get_from_gsheet(credentials, query, filename)                
    # elif source == 'sql':
    #     # Load data from SQL database
    #     print("Loading data from SQL database\n")
    #     data = get_from_sql(credentials, query, filename)
        
    return data

# Import data directly from google sheet or from a saved file
def load_utilities_data(
        credentials=None, query=None, directly=False, 
        filename='utilities.tsv'):    
    if directly:
        # Loading directly from google sheet
        data = load_data('gsheet', filename, credentials, query)
        
        # Convert the data to a Pandas dataframe
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Save the data for subsequent analysis
        df.to_csv(filename, sep='\t', header=True, index=False)        
        print("="*60)
        print(
            f"Successfully retrieved {query['worksheet_name']} from " f"googlesheet and saved to {filename}\n"
        )
    
    # Load file directly from file
    data = load_data('file', filename, '', '')
    df = pd.DataFrame(data)        
    print("="*60)
    print(f"Successfully read utility file from {filename}\n")   

    # Grab just the measurement columns
    df = df[
        [
            'date', 'elec_measured', 'elec_time', 'credit_ghs', 'credit_kwh',
             'ac_use', 'wifi_measured', 'wifi_time', 'credit_gb'
        ]
    ]
    
    ## Remove all empty rows
    df = df.dropna(how='all')
    
    return df

# Create Dataframe
def create_dataframe(data, columns):
    df = pd.DataFrame(data, columns=columns)
    return df
    
# # Get data from SQL database
# def get_from_sql(credentials, query, filename):
#     # Check to make sure Credential keys are present
#     credential_keys = ['uname', 'database', 'password', 'host', 'port']
#     for key in credential_keys:
#         if not credentials.get(key):
#             print(
#                 f"I'm missing {key} in the credentials dictionary. " 
#                 f"Quitting.\n"
#             )
#             sys.exit(1)

#     # Get database information
#     uname = credentials['uname']
#     database = credentials['database']
#     password = credentials['password']
#     host = credentials['host']
#     port = credentials['port']

#     # Establish Connection
#     conn = ps.connect(
#         database=database, user=uname, password=password, host=host, port=port
#     )

#     # setting auto commit false
#     conn.autocommit = False
    
#     # Creating a cursor object using the cursor() method
#     cursor = conn.cursor()

#     # Retrieve Data
#     cursor.execute(query)
#     data = cursor.fetchall()

#     # commit changes in the database and close connection
#     conn.commit()
#     conn.close()
  
#     return data
