import pandas as pd

# Load the CSV files
watch_time_df = pd.read_csv("limited_engagement_2023_h2.csv")
imdb_df = pd.read_csv("raw/imdb_tv_show_rating.xls")

# Function to find the first watch time title that contains the IMDb title
def find_matching_title(imdb_title, watch_titles):
    for watch_title in watch_titles:
        if imdb_title in watch_title.lower():
            return watch_title
    return None  # No match found

# Apply the function to match IMDb titles with watch time titles
imdb_df["matched_title"] = imdb_df["Title"].apply(lambda x: find_matching_title(x.lower(), watch_time_df["Title"]))

# Drop rows where no match was found
imdb_df = imdb_df.dropna(subset=["matched_title"])

# Merge based on matched titles
merged_df = watch_time_df.merge(imdb_df[['matched_title', 'Rating', 'Votes', "Genres"]], 
                                left_on="Title", right_on="matched_title", how="left")

# Convert 'genre' (comma-separated string) into a list
merged_df["genre_list"] = merged_df["Genres"].apply(lambda x: x.split(', ') if pd.notna(x) else [])

# Drop the extra column
merged_df.drop(columns=["matched_title"], inplace=True)

# Save the result
merged_df.to_csv("limited_engagement_2023_h2_plus.csv", index=False)
