import pandas as pd

# Load validation CSVs
artist_df = pd.read_csv('artist_val.csv', header=None, names=['image_path', 'artist_label'])
genre_df = pd.read_csv('genre_val.csv', header=None, names=['image_path', 'genre_label'])
style_df = pd.read_csv('style_val.csv', header=None, names=['image_path', 'style_label'])

print("Before merge:")
print(f"  Artist: {len(artist_df):,} rows")
print(f"  Genre:  {len(genre_df):,} rows")
print(f"  Style:  {len(style_df):,} rows")

# Check unique images
artist_imgs = set(artist_df['image_path'])
genre_imgs = set(genre_df['image_path'])
style_imgs = set(style_df['image_path'])

common = artist_imgs & genre_imgs & style_imgs
print(f"\nNo of images present in all 3 CSV files: {len(common):,}")

# Merge
merged = artist_df.merge(genre_df, on='image_path', how='inner') \
                 .merge(style_df, on='image_path', how='inner')

print(f"\nAfter merge: {len(merged):,} images")

# Save
merged[['image_path', 'artist_label', 'genre_label', 'style_label']].to_csv(
    'val_labels_merged.csv',
    index=False,
    header=False
)

print(f"\nSaved to val_labels_merged.csv")
print(f"  Format: image_path, artist_label, genre_label, style_label")

print("\nFirst 3 rows:")
print(merged.head(3))