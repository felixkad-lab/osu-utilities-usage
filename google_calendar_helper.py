from google.oauth2 import service_account
from googleapiclient.discovery import build
#from googleapiclient.errors import HttpError

# Initialize Google Calendar api
def initialize_gcal(json_keyfile):
    # Initialize 
    credentials = service_account.Credentials.from_service_account_file(
        json_keyfile, scopes=['https://www.googleapis.com/auth/calendar']
    )
    service = build('calendar', 'v3', credentials=credentials)
    return service

# Add event
def add_event_allday(
        calendar_id, summary_text, start_date, end_date, timezone, service):
    # Event Details
    event = {
        'summary': summary_text,
        'start': {
            'date': start_date,
            'timeZone': timezone,
        },
        'end': {
            'date': end_date,
            'time_zone': timezone,
        },
        'transparency': 'transparent', #this makes it an all-day event
    }

    # Insert the event into the calendar
    event = service.events().insert(
        calendarId=calendar_id, body=event
    ).execute()

    # Get event info
    event_link = event.get('htmlLink')
    eventID = event['id']
    print(f"Event with ID {eventID} added to Calendar successfully\n")
    return eventID, event_link

# Delete event                    
def delete_event(calendar_id, eventID, service):
    # Delete event from Google Calendar
    service.events().delete(calendarId=calendar_id, eventId=eventID).execute()
    print(f"Event with ID {eventID} deleted successfully\n")