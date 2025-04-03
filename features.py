import pandas as pd
from datetime import datetime, timedelta

# Load the dataset
df_global = pd.read_csv("global_clamped.csv")  # Adjust filename if needed

# Convert 'week' column to datetime format
df_global["week"] = pd.to_datetime(df_global["week"])

# Define the starting date of the dataset
start_date = datetime(2023, 1, 1)

def get_week_from_date(input_date):
    """Finds the start of the corresponding week for the given date."""
    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    
    # Find the start of the week (Sunday to Saturday format)
    week_start = start_date + timedelta(weeks=(input_date - start_date).days // 7)
    
    return week_start.strftime("%Y-%m-%d")

def get_most_popular_show(input_date):
    """Finds the most popular show for the week containing the given date."""
    week_start = get_week_from_date(input_date)

    # Filter data for the corresponding week
    weekly_data = df_global[df_global["week"] == week_start]

    # Get the most popular show (weekly_rank == 1)
    most_popular = weekly_data[weekly_data["weekly_rank"] == 1]

    if most_popular.empty:
        return f"No data available for the week starting {week_start}"

    return most_popular[["show_title", "category", "weekly_rank"]]

date_input = "2023-06-14"
print(get_most_popular_show(date_input))

df_countries = pd.read_csv("countries_clamped.csv")  

# Convert 'week' column to datetime format
df_countries["week"] = pd.to_datetime(df_countries["week"])

def get_most_popular_by_country(input_date, country_name):
    """Finds the most popular TV show and movie for a given week and country."""
    week_start = get_week_from_date(input_date)

    # Filter data for the corresponding week and country
    country_week_data = df_countries[(df_countries["week"] == week_start) & (df_countries["country_name"] == country_name)]

    if country_week_data.empty:
        return f"No data available for {country_name} in the week starting {week_start}"

    # Get the top-ranked movie and TV show
    top_movie = country_week_data[(country_week_data["category"] == "Films") & (country_week_data["weekly_rank"] == 1)]
    top_show = country_week_data[(country_week_data["category"] == "TV") & (country_week_data["weekly_rank"] == 1)]

    result = {}

    if not top_movie.empty:
        result["Most Popular Movie"] = top_movie.iloc[0]["show_title"]
    else:
        result["Most Popular Movie"] = "No movie found for this week"

    if not top_show.empty:
        result["Most Popular TV Show"] = top_show.iloc[0]["show_title"]
    else:
        result["Most Popular TV Show"] = "No TV show found for this week"

    return result

date_input = "2023-06-14"  
country_input = "United States"
print(get_most_popular_by_country(date_input, country_input))