import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(date):
    global_data = pd.read_csv("global_clamped.csv")
    country_data = pd.read_csv("countries_clamped.csv")
    watch_time_data = pd.read_csv("cleaned.csv")
    
    global_data['week'] = pd.to_datetime(global_data['week'])
    country_data['week'] = pd.to_datetime(country_data['week'])
    
    global_data = global_data.sort_values('week')
    country_data = country_data.sort_values('week')
    
    global_data = global_data[global_data['week'] < date]
    country_data = country_data[country_data['week'] < date]
    
    return global_data, country_data, watch_time_data

def preprocess_country_data(country_data, watch_time_data, selected_country):
    country_data = country_data[country_data['country_name'] == selected_country].copy()
    country_data['week_date'] = pd.to_datetime(country_data['week'])
    country_data['year'] = country_data['week_date'].dt.year
    country_data['month'] = country_data['week_date'].dt.month
    country_data['week_of_year'] = country_data['week_date'].dt.isocalendar().week
    
    movies_data = country_data[country_data['category'] == 'Films']
    tv_data = country_data[country_data['category'] == 'TV']
    
    movies_features = create_time_series_features(movies_data, watch_time_data)
    tv_features = create_time_series_features(tv_data, watch_time_data)
    
    return movies_features, tv_features, country_data

def create_time_series_features(data, watch_time_data):
    features = []
    titles = data['show_title'].unique()
    
    for title in titles:
        title_data = data[data['show_title'] == title].sort_values('week_date')
        if len(title_data) < 2:
            continue
        
        title_features = title_data.copy()
        title_features['prev_rank'] = title_features['weekly_rank'].shift(1)
        title_features['rank_change'] = title_features['weekly_rank'] - title_features['prev_rank']
        
        if 'weekly_hours_viewed' in title_features.columns:
            title_features['prev_hours_viewed'] = title_features['weekly_hours_viewed'].shift(1)
            title_features['hours_viewed_change'] = title_features['weekly_hours_viewed'] - title_features['prev_hours_viewed']
        
        watch_time_title = watch_time_data[watch_time_data['Title'] == title]
        if not watch_time_title.empty:
            title_features = pd.merge(title_features, 
                                      watch_time_title[['Title', 'Hours Viewed', 'Rating', 'Release Year']], 
                                      left_on='show_title', right_on='Title', how='left')
        
        title_features = title_features.dropna(subset=['prev_rank', 'weekly_rank'])
        features.append(title_features)
    
    return pd.concat(features, ignore_index=True) if features else pd.DataFrame()

def train_models(movies_features, tv_features):

    def train_random_forest(X, y):
        if X.empty:
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        return {'model': model, 'scaler': scaler, 'features': X.columns}
    
    x_movies, y_movies = prepare_model_data(movies_features)
    x_tv, y_tv = prepare_model_data(tv_features)
    return train_random_forest(x_movies, y_movies), train_random_forest(x_tv, y_tv)

def prepare_model_data(features):
    if features.empty:
        return pd.DataFrame(), None
    
    numeric_cols = ['weekly_rank', 'prev_rank', 'rank_change', 'month', 'week_of_year']
    
    if 'weekly_hours_viewed' in features.columns:
        numeric_cols.extend(['weekly_hours_viewed', 'prev_hours_viewed', 'hours_viewed_change'])
    
    if 'Rating' in features.columns:
        numeric_cols.append('Rating')
        
    available_cols = [col for col in numeric_cols if col in features.columns]
    X = features[available_cols].fillna(0)
    y = features['weekly_rank']
    return X, y

def predict_next_weeks(movies_model, tv_model, movies_features, tv_features, country_data):
    max_week = country_data['week_date'].max()
    next_week = max_week + timedelta(days=7)
    second_week = max_week + timedelta(days=14)
    movies_predictions = predict_category(movies_model, movies_features, 'Films', [next_week, second_week])
    tv_predictions = predict_category(tv_model, tv_features, 'TV', [next_week, second_week])
    
    return movies_predictions, tv_predictions


def predict_category(model, features, category, prediction_weeks):
    if model is None or features.empty:
        return pd.DataFrame()

    titles = features['show_title'].unique()
    all_predictions = []
    seen_titles = set()

    for week in prediction_weeks:
        week_predictions = []

        for title in titles:
            if title in seen_titles:
                continue 

            title_data = features[features['show_title'] == title].sort_values('week_date').tail(1)
            if title_data.empty:
                continue

            X_pred = pd.DataFrame({col: title_data[col].values[0] if col in model['features'] else 0 for col in model['features']}, index=[0])
            X_pred_scaled = model['scaler'].transform(X_pred)
            predicted_rank = model['model'].predict(X_pred_scaled)[0]

            week_predictions.append({'show_title': title, 'predicted_rank': predicted_rank, 'week': week, 'category': category})
            seen_titles.add(title)

        week_predictions.sort(key=lambda x: x['predicted_rank'])
        for i, pred in enumerate(week_predictions):
            pred['predicted_rank'] = i + 1

        all_predictions.extend(week_predictions[:10])  # Keep top 10 only

    return pd.DataFrame(all_predictions)


def visualize_predictions(movies_predictions, tv_predictions):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    if not movies_predictions.empty:
        next_week_movies = movies_predictions[movies_predictions['week'] == movies_predictions['week'].min()]
        next_week_movies = next_week_movies.sort_values('predicted_rank')
        
        ax1.barh(next_week_movies['show_title'][:10], 10 - next_week_movies['predicted_rank'][:10])
        ax1.set_title('Predicted Top 10 Movies for Next Week')
        ax1.set_xlabel('Rank Score (Higher is Better)')
        ax1.invert_yaxis()
    
    if not tv_predictions.empty:
        next_week_tv = tv_predictions[tv_predictions['week'] == tv_predictions['week'].min()]
        next_week_tv = next_week_tv.sort_values('predicted_rank')
        
        ax2.barh(next_week_tv['show_title'][:10], 10 - next_week_tv['predicted_rank'][:10])
        ax2.set_title('Predicted Top 10 TV Shows for Next Week')
        ax2.set_xlabel('Rank Score (Higher is Better)')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('next_week_predictions.png')
    plt.show()
    
    return fig

def predict_country_rank(selected_country, date):
    global_data, country_data, watch_time_data = load_data(date)
    
    movies_features, tv_features, country_data = preprocess_country_data(country_data, watch_time_data, selected_country)
    
    movies_model, tv_model = train_models(movies_features, tv_features)
    movies_predictions, tv_predictions = predict_next_weeks(movies_model, tv_model, movies_features, tv_features, country_data)
    
    print(f"\nPredicted Top 10 Movies for Next Week in {selected_country}:")
    print(movies_predictions.sort_values('predicted_rank').head(10)[['show_title', 'predicted_rank']])
    print(f"\nPredicted Top 10 TV Shows for Next Week in {selected_country}:")
    print(tv_predictions.sort_values('predicted_rank').head(10)[['show_title', 'predicted_rank']])
    
    return movies_predictions, tv_predictions

predict_country_rank('United States', '2023-04-04')
