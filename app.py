from flask import Flask, render_template, request
import pandas as pd
import datetime
import random

app = Flask(__name__)

# Sample parsing and prediction function
def parse_user_input(data):
    parsed_data = []
    for line in data.strip().split("\n"):
        try:
            # Split date and content
            date_content = line.split(" - ", 1)
            date_part = date_content[0]
            message_part = date_content[1].split(": ", 1)[1]

            # Extract time, event, and location
            time_event = message_part.split(" ", 1)
            event = time_event[1]
            time = time_event[0]
            location = "בחוץ" if "בחוץ" in event else "בבית"  # Example: infer location
            
            # Append parsed data
            parsed_data.append([date_part, time, event, location])
        except Exception as e:
            print(f"Error parsing line: {line} - {e}")
            continue
    return pd.DataFrame(parsed_data, columns=["Date", "Time", "Event", "Location"])

def predict_next_event(parsed_df):
    # Example logic for prediction based on averages
    if parsed_df.empty:
        return "No data provided. Cannot predict."

    # Extract hour of previous events
    parsed_df['Hour'] = pd.to_datetime(parsed_df['Time'], format='%H:%M').dt.hour
    next_hour = (parsed_df['Hour'].mean() + random.uniform(-1, 1)) % 24  # Add slight randomness
    next_time = f"{int(next_hour)}:{random.randint(0, 59):02d}"
    next_event = random.choice(['פיפי', 'קקי'])
    next_location = random.choice(parsed_df['Location'].unique())
    
    # Return formatted prediction
    return f"Next event: {next_event} at {next_time} in {next_location}"

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
