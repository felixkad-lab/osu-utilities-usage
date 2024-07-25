# I use this to provide information about json keyfile and google sheets

json_keyfile = (
    'REPLACE WITH GOOGLE ACCOUNT JSON KEYFILE LOCATION'
)

credentials = {'json_keyfile': json_keyfile}

query = {
    'spreadsheet_id': 'REPLACE WITH GOOGLE SHEET ID',
    'worksheet_name': 'NAME OF WORKSHEET',
    'range_name': 'RANGE TO IMPORT'
}

calendar_id = 'GOOGLE CALENDAR ID'