import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    global_data = pd.read_csv("global_clamped.csv")
    country_data = pd.read_csv("countries_clamped.csv")
    watch_time_data = pd.read_csv("cleaned.csv")
    return global_data, country_data, watch_time_data


def preprocess_data(global_data, country_data, watch_time_data):
    global_data['week_date'] = pd.to_datetime(global_data['week'])
    country_data['week_date'] = pd.to_datetime(country_data['week'])
    
    global_data['year'] = global_data['week_date'].dt.year
    global_data['month'] = global_data['week_date'].dt.month
    global_data['week_of_year'] = global_data['week_date'].dt.isocalendar().week
    
    movies_data = global_data[global_data['category'] == 'Films (English)']
    tv_data = global_data[global_data['category'] == 'TV (English)']
    
    movies_features = create_time_series_features(movies_data, country_data, watch_time_data)
    tv_features = create_time_series_features(tv_data, country_data, watch_time_data)
    
    return movies_features, tv_features

def create_time_series_features(data, country_data, watch_time_data):
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
        
        title_country_data = country_data[country_data['show_title'] == title]
        
        if not title_country_data.empty:
            country_counts = title_country_data.groupby('week')['country_name'].count().reset_index()
            country_counts.columns = ['week', 'country_count']
            title_features = pd.merge(title_features, country_counts, on='week', how='left')
            title_features['country_count'] = title_features['country_count'].fillna(0)
        
        else:
            title_features['country_count'] = 0
        
        if not watch_time_data.empty:
            watch_time_title = watch_time_data[watch_time_data['Title'] == title]
            if not watch_time_title.empty:
                title_features = pd.merge(title_features, 
                                          watch_time_title[['Title', 'Hours Viewed', 'Rating', 'Release Year']], 
                                          left_on='show_title', right_on='Title', how='left')
        
        title_features = title_features.dropna(subset=['prev_rank', 'weekly_rank'])
        
        features.append(title_features)
    
    if features:
        combined_features = pd.concat(features, ignore_index=True)
        return combined_features
    else:
        return pd.DataFrame()


def train_models(movies_features, tv_features):
    
    x_movies, y_movies = prepare_model_data(movies_features)
    if not x_movies.empty:
        movies_model = train_random_forest(x_movies, y_movies)
    else:
        movies_model = None
    
    x_tv, y_tv = prepare_model_data(tv_features)
    if not x_tv.empty:
        tv_model = train_random_forest(x_tv, y_tv)
    else:
        tv_model = None
    
    return movies_model, tv_model

def prepare_model_data(features):
    if features.empty:
        return pd.DataFrame(), None
    
    numeric_cols = ['weekly_rank', 'prev_rank', 'rank_change', 'cumulative_weeks_in_top_10', 
                    'country_count', 'month', 'week_of_year']
    
    if 'weekly_hours_viewed' in features.columns:
        numeric_cols.extend(['weekly_hours_viewed', 'prev_hours_viewed', 'hours_viewed_change'])
    if 'Rating' in features.columns:
        numeric_cols.append('Rating')
    if 'Hours Viewed' in features.columns:
        numeric_cols.append('Hours Viewed')
    
    available_cols = [col for col in numeric_cols if col in features.columns]
    
    X = features[available_cols].copy()
    y = features['weekly_rank']
    
    X = X.fillna(0)
    
    return X, y

def train_random_forest(X, y):
    if X.empty:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Model R² on training set: {train_score:.4f}")
    print(f"Model R² on test set: {test_score:.4f}")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance.head(10))
    
    return {'model': model, 'scaler': scaler, 'features': X.columns}


def predict_next_weeks(movies_model, tv_model, movies_features, tv_features, global_data):
    
    max_week = global_data['week_date'].max()
    next_week = max_week + timedelta(days=7)
    second_week = max_week + timedelta(days=14)
    
    prediction_weeks = [next_week, second_week]
    
    movies_predictions = predict_category(movies_model, movies_features, 'Films', prediction_weeks)
    tv_predictions = predict_category(tv_model, tv_features, 'TV', prediction_weeks)
    
    return movies_predictions, tv_predictions

def predict_category(model, features, category, prediction_weeks):
    if model is None or features.empty:
        return pd.DataFrame()
    
    titles = features['show_title'].unique()
    
    all_predictions = []
    
    for week in prediction_weeks:
        week_predictions = []
        
        for title in titles:
            
            title_data = features[features['show_title'] == title].sort_values('week_date').tail(1)
            
            if title_data.empty:
                continue
            
            X_pred = pd.DataFrame(index=[0])
            
            for col in model['features']:
                if col == 'weekly_rank':
                    X_pred[col] = title_data['weekly_rank'].values[0]
                elif col == 'prev_rank':
                    X_pred[col] = title_data['weekly_rank'].values[0]  
                elif col == 'rank_change':
                    X_pred[col] = 0  
                elif col == 'month':
                    X_pred[col] = week.month
                elif col == 'week_of_year':
                    X_pred[col] = week.isocalendar()[1]
                elif col in title_data.columns:
                    X_pred[col] = title_data[col].values[0]
                else:
                    X_pred[col] = 0
            
            X_pred_scaled = model['scaler'].transform(X_pred)
            predicted_rank = model['model'].predict(X_pred_scaled)[0]
            
            week_predictions.append({
                'show_title': title,
                'predicted_rank': predicted_rank,
                'week': week,
                'category': category,
                'current_rank': title_data['weekly_rank'].values[0] if 'weekly_rank' in title_data.columns else None,
                'weeks_in_top_10': title_data['cumulative_weeks_in_top_10'].values[0] if 'cumulative_weeks_in_top_10' in title_data.columns else None
            })
        
        week_predictions.sort(key=lambda x: x['predicted_rank'])
        
        top_10 = week_predictions[:10]
        all_predictions.extend(top_10)

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

def main():
    print("Loading data...")
    global_data, country_data, watch_time_data = load_data()
    
    print("Preprocessing data...")
    movies_features, tv_features = preprocess_data(global_data, country_data, watch_time_data)
    
    print("Training models...")
    movies_model, tv_model = train_models(movies_features, tv_features)
    
    print("Generating predictions...")
    movies_predictions, tv_predictions = predict_next_weeks(movies_model, tv_model, movies_features, tv_features, global_data)
    
    
    print("\nPredicted Top 10 Movies for Next Week:")
    next_week = movies_predictions['week'].max()
    print(movies_predictions[movies_predictions['week'] == next_week].sort_values('predicted_rank').head(10)[['show_title', 'predicted_rank']])
    
    print("\nPredicted Top 10 Movies for Second Week:")
    second_week = movies_predictions['week'].max()
    print(movies_predictions[movies_predictions['week'] == second_week].sort_values('predicted_rank').head(10)[['show_title', 'predicted_rank']])
    
    print("\nPredicted Top 10 TV Shows for Next Week:")
    next_week = tv_predictions['week'].max()
    print(tv_predictions[tv_predictions['week'] == next_week].sort_values('predicted_rank').head(10)[['show_title', 'predicted_rank']])
    
    print("\nPredicted Top 10 TV Shows for Second Week:")
    second_week = tv_predictions['week'].max()
    print(tv_predictions[tv_predictions['week'] == second_week].sort_values('predicted_rank').head(10)[['show_title', 'predicted_rank']])
    
    # visualize_predictions(movies_predictions, tv_predictions)
    
    return movies_predictions, tv_predictions