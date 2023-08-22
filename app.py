import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title('Movie Recommender')

if 'movies' not in st.session_state:
    st.session_state.movies = pd.read_csv('Movies_dataset.csv')

if 'watched_movies' not in st.session_state:
    wm = pd.read_csv('liked_movies.csv')
    st.session_state.watched_movies = wm

if 'temp_watched_movie_store' not in st.session_state:
    st.session_state.temp_watched_movie_store = ''

selected_genre = None # initialize so when someone clicks a genre it won't fail
lucky_response = ''

####################### pre-loads and constants ######################
df = st.session_state.movies

if 'clustered_movies' not in st.session_state:
    # transform "overview" column into vector
    vectorizer = TfidfVectorizer(stop_words='english')
    df['overview'] = df['overview'].fillna('')
    X = vectorizer.fit_transform(df['overview'])

    kmeans = KMeans(n_clusters=21, random_state=0).fit(X)
    df['cluster_label'] = kmeans.labels_
    st.session_state.clustered_movies = df

if 'tf_genres' not in st.session_state:
    feats = []

    for i in range(21):
        feats.append(
            [
                i, # the cluster number
                vectorizer.get_feature_names_out()[np.argsort(kmeans.cluster_centers_[i])[-6:]]
             ]
        )

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    df['pca_x_coordinates'] = X_pca[:, 0]
    df['pca_y_coordinates'] = X_pca[:, 1]

    st.session_state.tf_genres = feats


###################### ML stuff above ######################

def pick_proximal_movie(genre, movie):
    # pick a movie from the same cluster as the genre close to the pca_x_coordinate and pca_y_coordinate of a random
    # movie inside of the st.session_state.watched_movies dataframe.
    # get the cluster label of the genre
    genre_cluster_label = st.session_state.clustered_movies[st.session_state.clustered_movies['cluster_label'] == genre]


with st.container():

    st.write('What sort of movie are you in the mood for?')

    button_cols = 4
    cols = st.columns(button_cols)
    counter = 0
    for idx, genre in enumerate(st.session_state.tf_genres):
        if counter == 4:
            cols = st.columns(button_cols)
            counter = 0
        else:
            with cols[counter]:
                if st.button(label=' '.join(genre[1]), key=genre[0]):
                    selected_genre = genre[0]
                    random_genre_movie = \
                        st.session_state.clustered_movies[
                            st.session_state.clustered_movies['cluster_label'] == selected_genre]\
                            .sample(1)[['title', 'overview']]
                counter += 1

    if selected_genre is not None:
        st.write('Based on your current mood, would you want to watch "', random_genre_movie['title'].values[0], '"?')
        st.write(random_genre_movie['overview'].values[0])
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            if st.button('Yes!'):
                st.session_state.temp_watched_movie_store = random_genre_movie
        with u_col2:
            if st.button('No'):
                st.write("Sorry, try again!")

    st.divider()

    st.write(lucky_response)
    if st.button("I'm feeling lucky! Pick something random!"):
        movie = st.session_state.movies.sample(1)
        title = movie['title'].values[0]
        overview = movie['overview'].values[0]
        st.write("Would you like to watch ", title,'?')
        st.write("Here's a description: ", overview, '.')
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button('Yes, thanks!', key='yes_button'):
                lucky_response = "Great! Enjoy your movie!"
        with b_col2:
            if st.button('No, thanks.', key='no_button'):
                lucky_response = "Sorry, try again or select from the above!"
                st.write("Sorry, try again!")

    st.divider()

    if st.session_state.temp_watched_movie_store != '':
        print('firing')

        st.session_state.watched_movies.append(
            st.session_state.temp_watched_movie_store,
            ignore_index=True
        )
        st.session_state.temp_watched_movie_store = ''
    st.write("You've watched the following movies. These will be used as a basis for future recommendations.")
    st.write(st.session_state.watched_movies)
    st.write(st.session_state.temp_watched_movie_store)
