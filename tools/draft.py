def trim_dataset(df_train, max_per_genre=20000, preserve_labels = []):
  
  genre_cols = df_train.columns[1:]
  genre_counter = {genre: 0 for genre in genre_cols}

  df_shuffled = df_train.sample(fraction=1.0, shuffle=True, seed=42)
  kept_rows = []

  for row in df_shuffled.iter_rows(named=True):
      genres = [genre for genre in genre_cols if row[genre] == 1]
      if any(label in preserve_labels for label in genres):
          kept_rows.append(row)
          for genre in genres:
              genre_counter[genre] += 1


  for row in df_shuffled.iter_rows(named=True):
      genres = [genre for genre in genre_cols if row[genre] == 1]
      if row in kept_rows:
          continue
      
      if any(label in preserve_labels for label in genres):
          continue
      
      if all(genre_counter[genre] < max_per_genre for genre in genres):
          kept_rows.append(row)
          for genre in genres:
              genre_counter[genre] += 1

  df_trimmed = pl.DataFrame(kept_rows)
  return df_trimmed

trim_dataset(df_train, max_per_genre = 20000, preserve_labels=['War', 'History', 'Western'])