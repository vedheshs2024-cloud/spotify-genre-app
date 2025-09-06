import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 

st.set_page_config(page_title = "Minor Project")
st.header("Spotify Songs’ Genre Segmentation")

st.text("Jupyter NoteBook Link ")
st.link_button("Open Jupyter program", "https://drive.google.com/file/d/1E2HCT3EHcY4Jf4TXNHBplWu8heuSGDcF/view?usp=drivesdk")

df = pd.read_csv("spotify dataset.csv")
df.dropna(inplace=True)

df.drop_duplicates(subset=['track_name',  'track_artist'], keep='first', inplace=True)


df.drop(columns=["track_id","track_album_id","playlist_id"],inplace=True)

df.dropna(inplace=True)

st.subheader("spotify dataset")
st.dataframe(df)


st.write("The music recommendations made by Spotify, a music app, are excellent. It recommends music based on the songs and artists you usually listen to. The algorithm groups comparable features into clusters, and these clusters aid in comprehending the auditory properties of diverse songs. Use this specific data set to construct an automated system.")

cor=df.select_dtypes(include=["int","float"]).corr()
fig = plt.figure(figsize=(12,10))
sns.heatmap(cor,annot=True,cmap="coolwarm",linewidths=0.5)
st.subheader("---Correlation Matrix---")
st.pyplot(fig)

st.write("The graph visually represents the strength and direction of linear relationships between various audio features, with colors and numerical values indicating positive, negative, or weak correlations between pairs of variables. ")


fg = plt.figure(figsize=(12,10))
sns.countplot(x="playlist_genre",data=df,hue="playlist_genre")
st.subheader("---CountPlot Playlist genres---")
st.pyplot(fg)

st.latex("Interpretation:")
st.write("Dominant Genres: The longest bars indicate the most prevalent genres. In this chart, edm appears to have the highest count, followed by rap and pop.")
st.write("Less Frequent Genres: Shorter bars signify genres with lower counts. Rock, Mixtape, and Latin have comparatively lower counts than the top genres.")
st.write("Comparison: The chart allows for a quick visual comparison of the popularity or presence of different music genres within the dataset. For example, edm is significantly more frequent than rock")

X = df.select_dtypes(np.number)


st.header("Recommend Songs:")

cluster_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=5, random_state=42))
    ])
cluster_pipeline.fit(X)
cluster_labels = cluster_pipeline.predict(X)
df['cluster'] = cluster_labels



from sklearn.neighbors import NearestNeighbors

s = st.text_input("Enter the track_name : ")
a = st.text_input("Enter the track_artist : ")



f = st.button("submit")
if f:
    def recommend_songs(song_title,artist, features_df, original_df, n_recommendations=20):
        model = NearestNeighbors(n_neighbors=n_recommendations + 1)  # +1 to include the song itself
        model.fit(features_df)

        song_index_list = original_df.loc[
            (original_df["track_name"] == song_title) & (original_df["track_artist"] == artist)].index

        if song_index_list.empty:
            st.write("❌ Error: Song not found in the dataset.")
            return None

        song_index = song_index_list[0]
        
        distances, indices = model.kneighbors([features_df.iloc[song_index]])
        
        st.title("🎵 Selected Song:")
        st.dataframe(original_df.iloc[song_index])
        
        st.title("\n🎧 Recommended Songs:")
        st.dataframe(original_df.iloc[indices[0][1:]])


recommend_songs(s,a,X,df)






