import streamlit as st
import pandas as pd
from datetime import datetime
from global_viewcount_prediction import *
from countries_rank_prediction import *

COUNTRIES = ['Argentina', 'Australia', 'Austria', 'Bahamas', 'Bahrain', 'Bangladesh',
            'Belgium', 'Bolivia', 'Brazil', 'Bulgaria', 'Canada', 'Chile', 'Colombia',
            'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
            'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Finland',
            'France', 'Germany', 'Greece', 'Guadeloupe', 'Guatemala', 'Honduras',
            'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel',
            'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kenya', 'Kuwait', 'Latvia', 'Lebanon',
            'Lithuania', 'Luxembourg', 'Malaysia', 'Maldives', 'Malta', 'Martinique',
            'Mauritius', 'Mexico', 'Morocco', 'Netherlands', 'New Caledonia',
            'New Zealand', 'Nicaragua', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Panama',
            'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania',
            'Réunion', 'Saudi Arabia', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia',
            'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland',
            'Taiwan', 'Thailand', 'Trinidad and Tobago', 'Turkey', 'Ukraine',
            'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay',
            'Venezuela', 'Vietnam']

DATES = ["2023-01-01", "2023-06-25"]

def predict_top_10(date, country):
    if pd.to_datetime(date) < pd.to_datetime(DATES[0]) :
        return pd.DataFrame(), pd.DataFrame()

    if country.lower() == 'global' or country == '':
        movies_predictions, tv_predictions = main(date)
        return pd.DataFrame(movies_predictions), pd.DataFrame(tv_predictions)
    
    elif country not in COUNTRIES:
        print("Country not found in the current dataset.")
        return pd.DataFrame(), pd.DataFrame()
    
    else:
        movies_predictions, tv_predictions = predict_country_rank(country, date)
        return pd.DataFrame(movies_predictions), pd.DataFrame(tv_predictions)

st.title("Netflix Top 10 Rank Predictor")

date_string = st.text_input("Enter Date (YYYY-MM-DD)", datetime.today().strftime("%Y-%m-%d"))

try:
    date = datetime.strptime(date_string, "%Y-%m-%d")
except ValueError:
    st.error("Invalid date format. Please enter in YYYY-MM-DD format.")
    date = None  # Handle the case where the input is invalid

print(date)

country = st.text_input("Enter Country", "United States")

if st.button("Predict Top 10"):
    predictions_movie, predictions_tv = predict_top_10(date, country)

    if predictions_movie.empty or predictions_tv.empty:
        st.write("Date out of range. Please retry")

    else:
        st.write("### Predicted Top 10 Shows/Movies")
        st.dataframe(predictions_movie)
        st.dataframe(predictions_tv)

        # Visualization
        if country == "global" or country == "Global" or country == '':
            st.write("### Predicted Ranks")
            fig, ax = plt.subplots()
            ax.barh(predictions_movie["show_title"], predictions_movie["predicted_hours"], color="skyblue")
            ax.set_xlabel("Predicted Hours")
            ax.set_ylabel("Show Title")
            ax.invert_yaxis()
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.barh(predictions_tv["show_title"], predictions_tv["predicted_hours"], color="skyblue")
            ax.set_xlabel("Predicted Hours")
            ax.set_ylabel("Show Title")
            ax.invert_yaxis()
            st.pyplot(fig)

        else:
            st.write("### Predicted Ranks")
            fig, ax = plt.subplots()
            ax.barh(predictions_movie["show_title"], predictions_movie["predicted_rank"], color="skyblue")
            ax.set_xlabel("Predicted Rank")
            ax.set_ylabel("Show Title")
            ax.invert_yaxis()
            st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.barh(predictions_tv["show_title"], predictions_tv["predicted_rank"], color="skyblue")
            ax.set_xlabel("Predicted Rank")
            ax.set_ylabel("Show Title")
            ax.invert_yaxis() 
            st.pyplot(fig)