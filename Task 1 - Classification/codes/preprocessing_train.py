import pandas as pd

#we merge all 3 csv files into one based on common entries.
artist_df =pd.read_csv('artist_train.csv', header=None, names=['image_path', 'artist_label'])
genre_df= pd.read_csv('genre_train.csv', header=None, names=['image_path', 'genre_label'])
style_df= pd.read_csv('style_train.csv', header=None, names=['image_path', 'style_label'])

print("Before merge:")
print(f"  Artist: {len(artist_df):,} rows")
print(f"  Genre:  {len(genre_df):,} rows")
print(f"  Style:  {len(style_df):,} rows")


artist_imgs = set(artist_df['image_path'])
genre_imgs = set(genre_df['image_path'])
style_imgs = set(style_df['image_path'])

common = artist_imgs & genre_imgs & style_imgs
print(f"\nNo of images present in all 3 CSV files: {len(common):,}")

# inner join
merged = artist_df.merge(genre_df, on='image_path', how='inner') \
                 .merge(style_df, on='image_path', how='inner')

print(f"\nAfter merge: {len(merged):,} images")

# save as new csv file
merged[['image_path', 'artist_label', 'genre_label', 'style_label']].to_csv(
    'train_labels_merged.csv',
    index=False,
    header=False
)

print(f"\n Saved to train_labels_merged.csv")
print(f"  Format: image_path, artist_label, genre_label, style_label")

# Show first few rows
print("\nFirst 3 rows:")
print(merged.head(3))