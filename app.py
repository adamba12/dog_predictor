from flask import Flask, render_template, request
import pandas as pd
import datetime
import re
import random
import numpy as np  # Make sure numpy is imported for time difference calculations


app = Flask(__name__)

# Sample parsing and prediction function

def parse_user_input(data):
    parsed_data = []
    for line in data.strip().split("\n"):
        try:
            # Skip irrelevant messages such as group creation, encryption info, and deleted messages
            if ("Messages and calls are end-to-end encrypted" in line or 
                "You created this group" in line or 
                "You removed" in line or 
                "You deleted this message" in line):
                continue

            # Split date and content
            date_content = line.split(" - ", 1)
            if len(date_content) < 2:
                continue  # Skip invalid lines
            
            # Extract the date and remove time from it
            date_part = date_content[0].split(",")[0]  # Extract only the date (before the comma)
            message_part = date_content[1].split(": ", 1)[1] if len(date_content[1].split(": ", 1)) > 1 else ""

            # Remove "<This message was edited>" part if present
            message_part = message_part.replace("<This message was edited>", "").strip()

            # Extract time, event, and location
            time_event = message_part.split(" ", 1)
            time = time_event[0]  # Time is at the start
            event = time_event[1] if len(time_event) > 1 else ""  # The rest is the event

            # Check if the message contains any of the locations
            location_match = re.search(r"(בחוץ|במשרד|בבית)", event)
            if location_match:
                location = location_match.group(0)
                event = event.replace(location, '').strip()  # Remove location from event
            else:
                location = "Unknown"  # Default location if not found

            # Append parsed data
            parsed_data.append([date_part, time, event, location])
        except Exception as e:
            print(f"Error parsing line: {line} - {e}")
            continue
    return pd.DataFrame(parsed_data, columns=["Date", "Time", "Event", "Location"])



def predict_next_event(parsed_df):
    # Check if data is provided
    if parsed_df.empty:
        return "No data provided. Cannot predict."

    # 1. Predict Event Type based on Frequency
    event_counts = parsed_df['Event'].value_counts()
    next_event = event_counts.idxmax()  # Predict the most frequent event type (e.g., "פיפי" or "קקי")

    # 2. Predict Location based on Frequency
    location_counts = parsed_df['Location'].value_counts()
    next_location = location_counts.idxmax()  # Predict the most frequent location (e.g., "בחוץ" or "בבית")

    # 3. Predict Time based on Recent Data (reduce the prediction window)
    try:
        # Convert the 'Time' column to datetime format, then extract only the time part (HH:MM)
        parsed_df['Time'] = pd.to_datetime(parsed_df['Time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M')

        # Drop rows with invalid time values (NaT)
        parsed_df = parsed_df.dropna(subset=['Time'])

        # Check if there is any data after dropping NaT values
        if parsed_df.empty:
            return "No valid time data available for prediction."

        # Convert time to minutes and calculate the average time (based on more recent events)
        parsed_df['Time_in_minutes'] = parsed_df['Time'].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))

        # Use the last 3 events for prediction (to weight recent data more)
        recent_data = parsed_df.tail(3)

        # Calculate the average time of the last 3 events
        avg_time_in_minutes = recent_data['Time_in_minutes'].mean()

        # Predict the next time based on the average time with some randomness added to simulate variability
        time_variability = random.randint(-15, 15)  # Allow a random variation of up to 15 minutes
        predicted_time_in_minutes = int(avg_time_in_minutes) + time_variability
        predicted_time_in_minutes = max(0, min(1440, predicted_time_in_minutes))  # Ensure the time is between 00:00 and 23:59
        
        # Convert the predicted time back to hours and minutes
        predicted_hour = predicted_time_in_minutes // 60
        predicted_minute = predicted_time_in_minutes % 60
        next_time = f"{predicted_hour:02d}:{predicted_minute:02d}"

    except Exception as e:
        print(f"Error while processing time: {e}")
        return "Error processing time data"

    # 4. Determine if the prediction is for today or the next day based on the time of day
    current_date = datetime.datetime.now()

    # If the predicted time is after 22:00, shift to the next day
    predicted_time_obj = current_date.replace(hour=int(next_time.split(":")[0]), minute=int(next_time.split(":")[1]), second=0)

    # If the predicted time is after 22:00, the prediction should be for the next day
    if predicted_time_obj.hour >= 22:
        next_date_obj = current_date + datetime.timedelta(days=1)
        next_date_str = next_date_obj.strftime('%d/%m/%Y')
        next_day_name = next_date_obj.strftime('%A')
    else:
        # Otherwise, stay on the current day
        next_date_str = current_date.strftime('%d/%m/%Y')
        next_day_name = current_date.strftime('%A')

    # Ensure the predicted time is in the future
    # If the predicted time has already passed today, move it to the next available time
    if predicted_time_obj < current_date:
        predicted_time_obj += datetime.timedelta(hours=1)  # Add 1 hour to the prediction

    # Update the time based on this adjustment
    next_time = predicted_time_obj.strftime('%H:%M')

    # Return the prediction
    return f"Next event: {next_event} at {next_time} on {next_day_name}, {next_date_str} in {next_location}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    parsed_data_table = None

    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form['data']
        
        # Parse the user input
        parsed_df = parse_user_input(user_input)

        # Generate prediction based on parsed data
        prediction = predict_next_event(parsed_df)

        # Create an HTML table for the first 5 rows of parsed data
        parsed_data_table = parsed_df.head().to_html(classes='data', index=False)

    return render_template('index.html', data=parsed_data_table, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
