from flask import Flask, render_template, request
import datetime
import re
import numpy as np
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

            # Append parsed data as a dictionary
            parsed_data.append({
                "Date": date_part,
                "Time": time,
                "Event": event,
                "Location": location
            })
        except Exception as e:
            print(f"Error parsing line: {line} - {e}")
            continue
    return parsed_data  # Return list of dictionaries

def predict_next_event(parsed_data, event_type):
    if not parsed_data:
        return f"No data provided for {event_type}. Cannot predict.", None

    # Extract the latest event's timestamp
    latest_timestamp_str = parsed_data[-1].get('Date') + ' ' + parsed_data[-1].get('Time')
    latest_timestamp = datetime.datetime.strptime(latest_timestamp_str, "%d/%m/%Y %H:%M")
    current_date = datetime.datetime.now()

    # Calculate if the data is outdated
    is_outdated = (current_date - latest_timestamp).days > 0 or (current_date - latest_timestamp).seconds > 36000  # Data older than 10 hours

    # Filter data for the event type, also considering combined events like "פיפי וקקי" or "קקי פיפי"
    filtered_data = [entry for entry in parsed_data if event_type in entry.get('Event', '') or ("פיפי" in entry.get('Event', '') and "קקי" in entry.get('Event', ''))]
    if not filtered_data:
        return f"No valid {event_type} events found.", None

    # Predict Location Based on Frequency
    location_counts = {}
    for entry in filtered_data:
        location = entry.get('Location', "Unknown")
        location_counts[location] = location_counts.get(location, 0) + 1

    next_location = max(location_counts, key=location_counts.get) if location_counts else "Unknown"

    # Extract timestamps
    timestamps = []
    for entry in filtered_data:
        time_str = entry.get('Time', "")
        date_str = entry.get('Date', "")
        if ":" in time_str and date_str:
            time_obj = datetime.datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            timestamps.append(time_obj)

    if len(timestamps) < 2:
        return "Not enough data to make a prediction.", None

    # Sort timestamps
    timestamps = sorted(timestamps)
    time_differences = [(timestamps[i] - timestamps[i - 1]).total_seconds() / 60 for i in range(1, len(timestamps))]

    # Step 1: Interval-Based Prediction
    avg_interval = sum(time_differences) / len(time_differences)
    baseline_prediction_time = timestamps[-1] + datetime.timedelta(minutes=avg_interval)

    # Step 2: Exponential Smoothing for Trends
    if len(time_differences) > 2:
        smoothing_model = ExponentialSmoothing(time_differences, trend='add', seasonal=None, damped_trend=False)
        smoothing_fit = smoothing_model.fit()
        smoothed_interval = smoothing_fit.forecast(steps=1)[0]
    else:
        smoothed_interval = avg_interval  # Fallback to the average

    smoothing_prediction_time = timestamps[-1] + datetime.timedelta(minutes=smoothed_interval)

    # Step 3: Neural Network for Advanced Prediction
    if len(time_differences) >= 5:
        # Prepare data for neural network
        X = np.array(range(len(time_differences))).reshape(-1, 1)  # Index as feature
        y = np.array(time_differences)  # Intervals as target

        # Train neural network
        nn_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        nn_model.fit(X, y)

        # Predict next interval
        next_interval = nn_model.predict([[len(time_differences)]])[0]
    else:
        next_interval = smoothed_interval  # Fallback to smoothing prediction

    nn_prediction_time = timestamps[-1] + datetime.timedelta(minutes=next_interval)

    # Combine Predictions: Weighted Average
    baseline_seconds = (baseline_prediction_time - timestamps[-1]).total_seconds()
    smoothing_seconds = (smoothing_prediction_time - timestamps[-1]).total_seconds()
    nn_seconds = (nn_prediction_time - timestamps[-1]).total_seconds()

    combined_seconds = (
        0.5 * nn_seconds +
        0.3 * smoothing_seconds +
        0.2 * baseline_seconds
    )

    final_prediction_time = timestamps[-1] + datetime.timedelta(seconds=combined_seconds)

    # Ensure the final prediction is in the future by adjusting it if it's in the past
    if final_prediction_time < current_date:
        final_prediction_time = current_date + datetime.timedelta(minutes=avg_interval)  # Adjust to a time in the future

    # Return Prediction and Last Timestamp, include "outdated" warning
    prediction = f"Next event {event_type} will be at {final_prediction_time.strftime('%H:%M')} on {final_prediction_time.strftime('%A')}, {final_prediction_time.strftime('%d/%m/%Y')} in {next_location}"

    return prediction, latest_timestamp if is_outdated else None



@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_pipi = None
    prediction_kaki = None
    combined_prediction = None
    comment = None
    last_pipi = None
    last_kaki = None

    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form['data']
        
        # Parse the user input
        parsed_data = parse_user_input(user_input)

        # Generate predictions for "פיפי" and "קקי"
        prediction_pipi, last_pipi = predict_next_event(parsed_data, "פיפי")
        prediction_kaki, last_kaki = predict_next_event(parsed_data, "קקי")

        # Check if the input data is outdated (more than 15 hours)
        now = datetime.datetime.now()
        
        if (last_pipi and (now - last_pipi).total_seconds() > 54000) or (last_kaki and (now - last_kaki).total_seconds() > 54000):  # 15 hours
            comment = "Warning: Data is outdated (more than 15 hours since last event)."
        
        # Check if the times are within 20 minutes of each other
        if prediction_pipi != "No data provided. Cannot predict." and prediction_kaki != "No data provided. Cannot predict.":
            try:
                time_pipi = datetime.datetime.strptime(prediction_pipi.split(' at ')[1].split(' on ')[0], "%H:%M")
                time_kaki = datetime.datetime.strptime(prediction_kaki.split(' at ')[1].split(' on ')[0], "%H:%M")
                
                # Calculate time difference in minutes
                time_diff = abs((time_kaki - time_pipi).total_seconds() / 60)
                
                if time_diff <= 20:
                    combined_prediction = f"Next event פיפי וגם קקי will be at {time_pipi.strftime('%H:%M')} on {prediction_pipi.split(' on ')[1]}"
                    comment = "Note: Events predicted to be within 20 minutes from one another."
                    prediction_pipi = None  # Hide individual predictions if combined
                    prediction_kaki = None  # Hide individual predictions if combined
            except Exception as e:
                print(f"Error checking times: {e}")

    return render_template('index.html', prediction_pipi=prediction_pipi, prediction_kaki=prediction_kaki, combined_prediction=combined_prediction, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)
