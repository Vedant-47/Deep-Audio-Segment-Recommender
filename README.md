Deep Audio Segment Recommender

This project is a deep learning-based music recommendation system that uses audio features to identify genres and generate embeddings for similarity-based recommendations. It is built with TensorFlow/Keras, Librosa for audio processing, and Dash for interactive visualization.

Features:

~ Loads the GTZAN dataset (10 genres, 1000 audio tracks) (In future can add any music tracks).

~ Extracts mel-spectrogram features from audio.
  
    (A spectrogram is a visual representation of sound that shows how frequencies change over time.
  
       ~ The x-axis = time
   
       ~ The y-axis = frequency

       ~ The color/intensity = amplitude (loudness) of that frequency at that time)
   
~ Trains a CNN-based classifier to predict genres.

~ Generates embeddings for similarity search between tracks.

~ Provides an interactive Dash web app for exploring embeddings and recommendations.


Project Structure:

    Deep-Audio-Segment-Recommender/

      ├── dash_app.py              # Dash web application

      ├── trained_backend.py       # Training script for CNN model

      ├── model_output/            # Saved model, embeddings, and metadata
  
          ├── genre_cnn.h5
  
          ├── embeddings.npy

          └── metadata.csv

      ├── gtzan/                   # GTZAN dataset (10 genres of audio files)


Install dependencies:

    ~ TensorFlow / Keras
    ~ Librosa
    ~ Numpy
    ~ Pandas
    ~ Scikit-learn
    ~ Plotly
    ~ Dash

--> Usage of the code:

1. Structure of the dataset:
   
        gtzan/

          blues/
          classical/
          country/
          disco/
          hiphop/
          jazz/
          metal/
          pop/
          reggae/
          rock/

3. Train the Model:

python trained_backend.py (in cmd under the project directory)

~ generates accuracy = 74% (by training 100 epochs)


This will generate:

model_output/genre_cnn.h5 – trained CNN model

model_output/embeddings.npy – embeddings for each track

model_output/metadata.csv – metadata including file paths and labels




4. Run the Dash App:

python dash_app.py (then open http://127.0.0.1:8050 in your browser.)

5. Future Improvements:

~ Add data augmentation (pitch shift, time stretch, noise).

~ Implement transfer learning using pretrained CNNs.

~ Extend recommendation logic with collaborative filtering or hybrid methods.

~ Deploy the Dash app on a cloud platform (Heroku, AWS, or Azure).

