# 📦 Import all needed libraries
import streamlit as st  # To create the web app
import requests  # To get data from the weather API
import pandas as pd  # To work with data easily
import numpy as np  # For number operations
from sklearn.model_selection import train_test_split  # To split data for training and testing
from sklearn.preprocessing import LabelEncoder  # To change text data into numbers
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Machine learning models
from sklearn.metrics import mean_squared_error, accuracy_score  # To check how good the model is
from datetime import datetime, timedelta  # To work with date and time
import pytz  # For timezone

# 🛠️ API setup
API_KEY = 'bdf53fb14cacdd1cc84c85ba4ccbe98c'  # Your weather API key
BASE_URL = 'https://api.openweathermap.org/data/2.5/'  # Base URL for API

# 📡 Function to get current weather
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'Wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed']
    }

# 📂 Function to read old weather data (csv file)
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()  # Remove missing values
    df = df.drop_duplicates()  # Remove same rows
    return df

# 🧹 Prepare data for rain prediction
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])  # Change text to number
    data['RainTomorrowk'] = le.fit_transform(data['RainTomorrow'])  # Change Yes/No to 1/0

    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']

    return X, y, le

# 🌧️ Train rain prediction model
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Accuracy of Rain Prediction Model:", accuracy_score(y_test, y_pred))

    return model

# 🧹 Prepare data for temperature and humidity future prediction
def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    return X, y

# 🔥 Train regression model
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# 🔮 Predict future values
def predict_future(model, current_value):
    predictions = [current_value]

    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])

    return predictions[1:]

# 🌟 Main Streamlit App
def main():
    # Set Background Image
    page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(0, 0, 128, 0.6), rgba(0, 0, 128, 0.6)), 
                      url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
    background-size: cover;
    background-position: center;
    color: white;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: white;
}
</style>
"""
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title('🌦️ Weather Forecast and Rain Prediction App')

    city = st.text_input('Enter City Name:', 'Kolkata')

    if st.button('Check Weather'):
        try:
            current_weather = get_current_weather(city)

            # Read historical data
            historical_data = read_historical_data('weather.csv')

            # Train Rain Prediction Model
            X, y, le = prepare_data(historical_data)
            rain_model = train_rain_model(X, y)

            # Wind Direction to Compass
            wind_deg = current_weather['Wind_gust_dir'] % 360
            compass_points = [
                ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
                ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
                ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
                ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
                ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
                ("NNW", 326.25, 348.75)
            ]
            compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)

            compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

            # Make current weather data ready for model
            current_data = {
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'WindGustDir': compass_direction_encoded,
                'WindGustSpeed': current_weather['Wind_Gust_Speed'],
                'Humidity': current_weather['humidity'],
                'Pressure': current_weather['pressure'],
                'Temp': current_weather['current_temp'],
            }

            current_df = pd.DataFrame([current_data])

            # Predict rain
            rain_prediction = rain_model.predict(current_df)[0]

            # Predict future temperature and humidity
            X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
            X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

            temp_model = train_regression_model(X_temp, y_temp)
            hum_model = train_regression_model(X_hum, y_hum)

            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_humidity = predict_future(hum_model, current_weather['humidity'])

            # Future time setup
            timezone = pytz.timezone('Asia/Kolkata')
            now = datetime.now(timezone)
            next_hour = now + timedelta(hours=1)
            next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

            future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

            # Show Results
            st.header(f"📍 Weather Forecast for {city}, {current_weather['country']}")

            st.write(f"🌡️ Current Temp: {current_weather['current_temp']}°C")
            st.write(f"🤔 Feels Like: {current_weather['feels_like']}°C")
            st.write(f"⬇️ Min Temp: {current_weather['temp_min']}°C")
            st.write(f"⬆️ Max Temp: {current_weather['temp_max']}°C")
            st.write(f"💧 Humidity: {current_weather['humidity']}%")
            st.write(f"🌥️ Condition: {current_weather['description'].capitalize()}")
            st.write(f"🌧️ Rain Expected: {'Yes' if rain_prediction else 'No'}")

            st.subheader("Future Temperature Prediction (Next 5 Hours)")
            for time, temp in zip(future_times, future_temp):
                st.write(f"{time}: {round(temp, 1)}°C")

            st.subheader("Future Humidity Prediction (Next 5 Hours)")
            for time, hum in zip(future_times, future_humidity):
                st.write(f"{time}: {round(hum, 1)}%")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# 🏃‍♂️ Run the main function
if __name__ == '__main__':
    main()
