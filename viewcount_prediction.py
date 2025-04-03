import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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

    if 'weekly_hours_viewed' not in global_data.columns:
        print("Warning: No weekly_hours_viewed column found in global data.")
        return pd.DataFrame(), pd.DataFrame()
    

    global_data['year'] = global_data['week_date'].dt.year
    global_data['month'] = global_data['week_date'].dt.month
    global_data['week_of_year'] = global_data['week_date'].dt.isocalendar().week
    global_data['day_of_year'] = global_data['week_date'].dt.dayofyear

    movies_data = global_data[global_data['category'] == 'Films (English)']
    tv_data = global_data[global_data['category'] == 'TV (English)']

    movies_features = create_view_count_features(movies_data, country_data, watch_time_data)
    tv_features = create_view_count_features(tv_data, country_data, watch_time_data)
    
    return movies_features, tv_features

def create_view_count_features(data, country_data, watch_time_data):
    if 'weekly_hours_viewed' not in data.columns:
        return pd.DataFrame()
    
    features = []
    titles = data['show_title'].unique()
    
    for title in titles:
        title_data = data[data['show_title'] == title].sort_values('week_date')
        
        if len(title_data) < 2:
            continue

        title_features = title_data.copy()

        title_features['hours_viewed_lag1'] = title_features['weekly_hours_viewed'].shift(1)
        title_features['hours_viewed_lag2'] = title_features['weekly_hours_viewed'].shift(2)
        title_features['hours_viewed_lag3'] = title_features['weekly_hours_viewed'].shift(3)

        title_features['growth_rate'] = (title_features['weekly_hours_viewed'] / 
                                        title_features['hours_viewed_lag1'] - 1) * 100

        title_features['rank_lag1'] = title_features['weekly_rank'].shift(1)
        title_features['rank_change'] = title_features['weekly_rank'] - title_features['rank_lag1']
        
        title_country_data = country_data[country_data['show_title'] == title]
        if not title_country_data.empty:
            
            country_counts = title_country_data.groupby('week')['country_name'].count().reset_index()
            country_counts.columns = ['week', 'country_count']
            title_features = pd.merge(title_features, country_counts, on='week', how='left')
            title_features['country_count'] = title_features['country_count'].fillna(0)
            
            title_features['country_count_lag1'] = title_features['country_count'].shift(1)
            title_features['country_growth'] = title_features['country_count'] - title_features['country_count_lag1']
        else:
            title_features['country_count'] = 0
            title_features['country_count_lag1'] = 0
            title_features['country_growth'] = 0
        
        if not watch_time_data.empty:
            watch_time_title = watch_time_data[watch_time_data['Title'] == title]
            if not watch_time_title.empty:
                
                selected_columns = ['Title', 'Hours Viewed', 'Rating', 'Release Year', 'Genre']
                available_columns = [col for col in selected_columns if col in watch_time_title.columns]
                
                if available_columns:
                    title_features = pd.merge(title_features, 
                                             watch_time_title[available_columns], 
                                             left_on='show_title', right_on='Title', how='left')
        
        if 'Release Year' in title_features.columns:
            
            current_year = title_features['year'].max()
            title_features['years_since_release'] = current_year - title_features['Release Year']
            title_features['weeks_since_release'] = (title_features['years_since_release'] * 52 + 
                                                   title_features['week_of_year'])
        
        if len(title_features) >= 3:

            title_features['rolling_avg_hours'] = title_features['weekly_hours_viewed'].rolling(3).mean().shift(1)
            
            title_features['hours_growth_lag1'] = title_features['growth_rate'].shift(1)
            title_features['acceleration'] = title_features['growth_rate'] - title_features['hours_growth_lag1']
        
        title_features['quarter'] = title_features['month'].apply(lambda x: (x-1)//3 + 1)

        title_features = title_features.dropna(subset=['hours_viewed_lag1', 'weekly_hours_viewed'])
        
        features.append(title_features)
    
    if features:
        combined_features = pd.concat(features, ignore_index=True)
        return combined_features
    else:
        return pd.DataFrame()

def train_models(movies_features, tv_features):
    
    X_movies, y_movies = prepare_view_count_model_data(movies_features)
    if not X_movies.empty:
        movies_model = train_view_count_model(X_movies, y_movies)
    else:
        movies_model = None

    X_tv, y_tv = prepare_view_count_model_data(tv_features)
    if not X_tv.empty:
        tv_model = train_view_count_model(X_tv, y_tv)
    else:
        tv_model = None
    
    return movies_model, tv_model

def prepare_view_count_model_data(features):
    if features.empty:
        return pd.DataFrame(), None
    
    target_col = 'weekly_hours_viewed'
    
    potential_features = [
        'month', 'week_of_year', 'day_of_year', 'quarter',
        
        'hours_viewed_lag1', 'hours_viewed_lag2', 'hours_viewed_lag3',
        'rank_lag1', 'rank_change', 'growth_rate',
        'country_count', 'country_count_lag1', 'country_growth',
        'Rating', 'cumulative_weeks_in_top_10',

        'rolling_avg_hours', 'acceleration', 'years_since_release', 'weeks_since_release'
    ]

    available_cols = [col for col in potential_features if col in features.columns]
    
    X = features[available_cols].copy()
    y = features[target_col]

    X = X.fillna(0)
    
    return X, y

def train_view_count_model(X, y):
    if X.empty:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    

    
    

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_train_score = rf_model.score(X_train_scaled, y_train)
    rf_test_score = rf_model.score(X_test_scaled, y_test)
    
    

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_train_score = gb_model.score(X_train_scaled, y_train)
    gb_test_score = gb_model.score(X_test_scaled, y_test)
    
    print(f"Random Forest R² on training set: {rf_train_score:.4f}")
    print(f"Random Forest R² on test set: {rf_test_score:.4f}")
    print(f"Gradient Boosting R² on training set: {gb_train_score:.4f}")
    print(f"Gradient Boosting R² on test set: {gb_test_score:.4f}")
    
    

    if gb_test_score > rf_test_score:
        model = gb_model
        print("Using Gradient Boosting model")
    else:
        model = rf_model
        print("Using Random Forest model")
    
    

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance.head(10))
    
    return {'model': model, 'scaler': scaler, 'features': X.columns}

def predict_view_counts(movies_model, tv_model, movies_features, tv_features, global_data):
    

    max_week = global_data['week_date'].max()
    next_week = max_week + timedelta(days=7)
    second_week = max_week + timedelta(days=14)
    
    prediction_weeks = [next_week, second_week]
    
    movies_predictions = predict_category_view_counts(movies_model, movies_features, 'Films', prediction_weeks)
    tv_predictions = predict_category_view_counts(tv_model, tv_features, 'TV', prediction_weeks)
    
    return movies_predictions, tv_predictions

def predict_category_view_counts(model, features, category, prediction_weeks):
    if model is None or features.empty:
        return pd.DataFrame()
    
    

    titles = features['show_title'].unique()
    
    all_predictions = []
    
    for week_idx, week in enumerate(prediction_weeks):
        week_predictions = []
        
        for title in titles:
            

            title_data = features[features['show_title'] == title].sort_values('week_date').tail(3)
            
            if title_data.empty:
                continue
            
            

            X_pred = pd.DataFrame(index=[0])
            
            

            latest_data = title_data.iloc[-1]
            
            

            if week_idx == 0:
                

                for col in model['features']:
                    if col in latest_data:
                        X_pred[col] = latest_data[col]
                    elif col == 'month':
                        X_pred[col] = week.month
                    elif col == 'week_of_year':
                        X_pred[col] = week.isocalendar()[1]
                    elif col == 'day_of_year':
                        X_pred[col] = week.timetuple().tm_yday
                    elif col == 'quarter':
                        X_pred[col] = (week.month - 1) // 3 + 1
                    else:
                        X_pred[col] = 0
                
                

                if 'hours_viewed_lag1' in model['features']:
                    X_pred['hours_viewed_lag1'] = latest_data['weekly_hours_viewed']
                
                if 'hours_viewed_lag2' in model['features'] and len(title_data) >= 2:
                    X_pred['hours_viewed_lag2'] = title_data.iloc[-2]['weekly_hours_viewed']
                
                if 'hours_viewed_lag3' in model['features'] and len(title_data) >= 3:
                    X_pred['hours_viewed_lag3'] = title_data.iloc[-3]['weekly_hours_viewed']
            
            

            else:
                

                first_week_pred = next(
                    (p for p in all_predictions if p['show_title'] == title and p['week'] == prediction_weeks[0]),
                    None
                )
                
                

                if first_week_pred:
                    for col in model['features']:
                        if col == 'hours_viewed_lag1':
                            X_pred[col] = first_week_pred['predicted_hours']
                        elif col == 'hours_viewed_lag2':
                            X_pred[col] = latest_data['weekly_hours_viewed']
                        elif col == 'hours_viewed_lag3' and len(title_data) >= 2:
                            X_pred[col] = title_data.iloc[-2]['weekly_hours_viewed']
                        elif col == 'month':
                            X_pred[col] = week.month
                        elif col == 'week_of_year':
                            X_pred[col] = week.isocalendar()[1]
                        elif col == 'day_of_year':
                            X_pred[col] = week.timetuple().tm_yday
                        elif col == 'quarter':
                            X_pred[col] = (week.month - 1) // 3 + 1
                        elif col in latest_data:
                            X_pred[col] = latest_data[col]
                        else:
                            X_pred[col] = 0
                else:
                    

                    continue
            
            

            X_pred = X_pred.fillna(0)
            
            

            for col in model['features']:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            

            X_pred_scaled = model['scaler'].transform(X_pred)
            predicted_hours = max(0, model['model'].predict(X_pred_scaled)[0])  

            
            week_predictions.append({
                'show_title': title,
                'predicted_hours': predicted_hours,
                'week': week,
                'category': category,
                'current_hours': latest_data['weekly_hours_viewed'] if 'weekly_hours_viewed' in latest_data else None,
                'weeks_in_top_10': latest_data['cumulative_weeks_in_top_10'] if 'cumulative_weeks_in_top_10' in latest_data else None
            })
        
        

        week_predictions.sort(key=lambda x: x['predicted_hours'], reverse=True)
        
        

        top_10 = week_predictions[:10]
        
        

        all_predictions.extend(top_10)
    
    return pd.DataFrame(all_predictions)

def visualize_view_count_predictions(movies_predictions, tv_predictions):
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    

    if not movies_predictions.empty:
        next_week_movies = movies_predictions[movies_predictions['week'] == movies_predictions['week'].min()]
        next_week_movies = next_week_movies.sort_values('predicted_hours', ascending=False).head(10)
        
        

        next_week_movies['hours_in_millions'] = next_week_movies['predicted_hours'] / 1_000_000
        
        sns.barplot(x='hours_in_millions', y='show_title', data=next_week_movies, ax=ax1)
        ax1.set_title('Predicted Top 10 Movies for Next Week')
        ax1.set_xlabel('Predicted Hours Viewed (millions)')
        ax1.set_ylabel('')
    
    

    if not tv_predictions.empty:
        next_week_tv = tv_predictions[tv_predictions['week'] == tv_predictions['week'].min()]
        next_week_tv = next_week_tv.sort_values('predicted_hours', ascending=False).head(10)
        
        

        next_week_tv['hours_in_millions'] = next_week_tv['predicted_hours'] / 1_000_000
        
        sns.barplot(x='hours_in_millions', y='show_title', data=next_week_tv, ax=ax2)
        ax2.set_title('Predicted Top 10 TV Shows for Next Week')
        ax2.set_xlabel('Predicted Hours Viewed (millions)')
        ax2.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('next_week_view_predictions.png')
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
    movies_predictions, tv_predictions = predict_view_counts(movies_model, tv_model, movies_features, tv_features, global_data)
    
    

    print("\nPredicted Top 10 Movies for Next Week (by hours viewed):")
    if not movies_predictions.empty:
        next_week = movies_predictions['week'].min()
        movies_next_week = movies_predictions[movies_predictions['week'] == next_week].sort_values('predicted_hours', ascending=False).head(10)
        print(movies_next_week[['show_title', 'predicted_hours']])
        
        print("\nPredicted Top 10 Movies for Second Week (by hours viewed):")
        second_week = movies_predictions['week'].max()
        movies_second_week = movies_predictions[movies_predictions['week'] == second_week].sort_values('predicted_hours', ascending=False).head(10)
        print(movies_second_week[['show_title', 'predicted_hours']])
    else:
        print("No movie predictions generated. Check if weekly_hours_viewed data is available.")
    
    print("\nPredicted Top 10 TV Shows for Next Week (by hours viewed):")
    if not tv_predictions.empty:
        next_week = tv_predictions['week'].min()
        tv_next_week = tv_predictions[tv_predictions['week'] == next_week].sort_values('predicted_hours', ascending=False).head(10)
        print(tv_next_week[['show_title', 'predicted_hours']])
        
        print("\nPredicted Top 10 TV Shows for Second Week (by hours viewed):")
        second_week = tv_predictions['week'].max()
        tv_second_week = tv_predictions[tv_predictions['week'] == second_week].sort_values('predicted_hours', ascending=False).head(10)
        print(tv_second_week[['show_title', 'predicted_hours']])
    else:
        print("No TV show predictions generated. Check if weekly_hours_viewed data is available.")
    
    

    visualize_view_count_predictions(movies_predictions, tv_predictions)
    
    return movies_predictions, tv_predictions

if __name__ == "__main__":
    movies_predictions, tv_predictions = main()