from flask import Flask, render_template, request
import datetime
import pandas as pd
import random

app = Flask(__name__)

# Sample data: User should provide this or upload as CSV
event_data = [
    ['18/12/2023', '07:10', 'פיפי וקקי', 'בחוץ'],
    ['19/12/2023', '12:45', 'פיפי וקקי', 'בחוץ'],
    ['19/12/2023', '16:29', 'קקי פיפי', 'בחוץ'],
    ['20/12/2023', '07:30', 'פיפי', 'בחוץ'],
    ['20/12/2023', '18:12', 'פיפי', 'בחוץ']
]

# Convert to DataFrame for easier handling
df = pd.DataFrame(event_data, columns=["Date", "Time", "Event", "Location"])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Fetch user input data
        user_input = request.form['data']
        
        # Example process to handle the user input
        user_data = []
        for line in user_input.strip().split("\n"):
            parts = line.split(" ")
            date_time = parts[0].split(',')
            event = " ".join(parts[2:])
            date = date_time[0]
            time = date_time[1]
            location = parts[-1]
            user_data.append([date, time, event, location])
        
        # Create DataFrame for user input
        user_df = pd.DataFrame(user_data, columns=["Date", "Time", "Event", "Location"])

        # Predict the next event (just a placeholder random example)
        next_time = f"{random.randint(6, 10)}:{random.randint(10, 59):02d}"  # Example next time
        next_event = random.choice(['פיפי', 'קקי'])
        next_location = 'בחוץ'
        
        # Get current date, year, and day of the week
        current_datetime = datetime.datetime.now()
        current_year = current_datetime.year
        current_day = current_datetime.strftime('%A')  # Get day of the week
        full_date = current_datetime.strftime('%d/%m/%Y')  # Full date (DD/MM/YYYY)

        # Format prediction with full date and day of the week
        prediction = f"Next event: {next_event} at {next_time} on {current_day}, {full_date} {next_location}"

        # Pass the data to HTML template
        return render_template('index.html', data=user_df.to_html(classes='data'), prediction=prediction)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
