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

    # 3. Predict Time based on Historical Data
    try:
        # Convert the 'Time' column to datetime format, then extract only the time part (HH:MM)
        parsed_df['Time'] = pd.to_datetime(parsed_df['Time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M')

        # Drop rows with invalid time values (NaT)
        parsed_df = parsed_df.dropna(subset=['Time'])

        # Check if there is any data after dropping NaT values
        if parsed_df.empty:
            return "No valid time data available for prediction."

        # Convert time to minutes for easier calculation
        parsed_df['Time_in_minutes'] = parsed_df['Time'].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))

        # Check the data to make sure time conversion is correct
        print("Parsed Times:", parsed_df['Time_in_minutes'].values)

        # Calculate the time differences between the most recent events
        recent_times = parsed_df['Time_in_minutes'].tail(5).values
        if len(recent_times) < 2:
            return "Not enough data to calculate time differences."

        time_differences = np.diff(recent_times)
        print("Time Differences:", time_differences)

        # Calculate the average time difference
        avg_time_diff = np.mean(time_differences)
        print("Average Time Difference:", avg_time_diff)

        # Predict the next time based on the most recent time difference
        last_time = recent_times[-1]
        predicted_time_in_minutes = last_time + avg_time_diff

        # Ensure the time is within 24 hours
        predicted_time_in_minutes = max(0, min(1440, predicted_time_in_minutes))  # Ensure the time is between 00:00 and 23:59

        # Convert the predicted time back to hours and minutes, ensuring it is an integer
        predicted_hour = int(predicted_time_in_minutes // 60)  # Ensure it's an integer
        predicted_minute = int(predicted_time_in_minutes % 60)  # Ensure it's an integer
        next_time = f"{predicted_hour:02d}:{predicted_minute:02d}"

    except Exception as e:
        print(f"Error while processing time: {e}")
        return f"Error processing time data: {e}"

    # 4. Predict the Date (next day based on current date)
    current_date = datetime.datetime.now()
    next_date = current_date + datetime.timedelta(days=1)  # Predict for the next day
    next_date_str = next_date.strftime('%d/%m/%Y')
    next_day_name = next_date.strftime('%A')

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
