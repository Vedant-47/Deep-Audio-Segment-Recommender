"""
Dash app:
 - Loads embeddings.npy + metadata.csv + trained model (genre_cnn.h5)
 - Upload a song -> show waveform, mel-spectrogram, predicted genre, and top-k similar songs
 - Cytoscape graph to visualize query node + nearest neighbors (edge weight = similarity)
 - Playlist builder: user picks >=5 favorites, click "Recommend" -> show playlist (top N)
"""

import base64
import io
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
import dash
from dash import html, dcc, Input, Output, State
from dash import dash_table
import dash_cytoscape as cyto
import soundfile as sf
import tensorflow as tf

# -------- CONFIG --------
OUT_DIR = "model_output"
MODEL_PATH = OUT_DIR + "/genre_cnn.h5"
EMB_PATH = OUT_DIR + "/embeddings.npy"
META_PATH = OUT_DIR + "/metadata.csv"
IMG_SHAPE = (128, 128)
N_MELS = 128
TARGET_SR = 22050

# -------- LOAD ARTIFACTS --------
print("Loading artifacts...")
model = tf.keras.models.load_model(MODEL_PATH)
embed_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('embedding').output)
embeddings = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
print("Loaded model, embeddings, metadata.")


# -------- HELPERS --------
def load_audio_bytes(contents):
    # contents from dcc.Upload: "data:audio/wav;base64,...."
    header, b64 = contents.split(',', 1)
    data = base64.b64decode(b64)
    # read into numpy using soundfile
    audio_bytes = io.BytesIO(data)
    y, sr = librosa.load(audio_bytes, sr=TARGET_SR, mono=True, duration=30)
    if len(y) < TARGET_SR * 30:
        y = np.pad(y, (0, max(0, TARGET_SR * 30 - len(y))), mode='constant')
    return y, sr

def mel_image_from_audio(y, sr, n_mels=N_MELS, shape=IMG_SHAPE):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    Smin, Smax = S_db.min(), S_db.max()
    S_scaled = (S_db - Smin) / (Smax - Smin + 1e-6)
    img = np.flip(S_scaled, axis=0)
    img_resized = resize(img, shape, mode='reflect', anti_aliasing=True)
    return img_resized.astype(np.float32)

def array_to_base64_png(img):
    # img: 2D array
    fig, ax = plt.subplots(figsize=(6,3))
    ax.imshow(img, aspect='auto')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def predict_genre_and_embedding_from_audio(y, sr):
    img = mel_image_from_audio(y, sr)
    x = img[np.newaxis, ..., np.newaxis]
    pred_probs = model.predict(x)[0]
    pred_idx = np.argmax(pred_probs)
    genres = sorted(meta['genre'].unique())
    pred_genre = genres[pred_idx] if pred_idx < len(genres) else "Unknown"
    embedding = embed_model.predict(x)[0]
    return pred_genre, pred_probs, embedding, img

def top_k_similar(embedding, k=10):
    sims = cosine_similarity(embedding.reshape(1, -1), embeddings)[0]
    order = sims.argsort()[::-1]
    return order[:k], sims[order[:k]]

def build_cytoscape_elements(query_node, neighbors_idx, sims):
    elements = []
    # query node
    elements.append({'data': {'id': 'query', 'label': 'Query'}, 'classes': 'query'})
    # neighbor nodes
    for i, idx in enumerate(neighbors_idx):
        label = f"{meta.loc[idx, 'genre']} - {meta.loc[idx,'filepath'].split('/')[-1]}"
        elements.append({'data': {'id': f'n{i}', 'label': label, 'track_idx': int(idx)}})
        # edge with weight
        elements.append({'data': {'source': 'query', 'target': f'n{i}', 'weight': float(sims[i])}})
    return elements

# -------- APP UI --------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Music Analysis & Graph-based Recommender (GTZAN)"),
    dcc.Tabs([
        dcc.Tab(label='Upload & Analyze', children=[
            html.P("Upload an mp3/wav (<=30s):"),
            dcc.Upload(id='upload-audio', children=html.Div(['Drag and Drop or ', html.A('Select File')]), style={
                'width': '60%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px'
            }),
            html.Div(id='audio-preview'),
            html.Hr(),
            html.Div(id='pred-output'),
            html.H4("Similarity Graph (query node highlighted)"),
            cyto.Cytoscape(
                id='music-graph',
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '450px'},
                elements=[]
            ),
            html.Hr(),
            html.H4("Top similar tracks"),
            dash_table.DataTable(id='similar-table', columns=[
                {"name":"track_idx","id":"track_idx"},
                {"name":"filepath","id":"filepath"},
                {"name":"genre","id":"genre"},
                {"name":"similarity","id":"similarity"}
            ], page_size=10)
        ]),
        dcc.Tab(label='Playlist Builder', children=[
            html.P("Select favourite tracks (click rows below) - choose at least 5:"),
            dash_table.DataTable(
                id='all-tracks-table',
                columns=[{"name":"idx","id":"idx"}, {"name":"filepath","id":"filepath"}, {"name":"genre","id":"genre"}],
                data=[{"idx":int(i), "filepath": row.filepath, "genre": row.genre} for i,row in meta.iterrows()],
                page_size=10,
                row_selectable='multi'
            ),
            html.Br(),
            html.Button("Generate Playlist from selection", id='gen-playlist-btn', n_clicks=0),
            html.Div(id='playlist-output'),
            html.Hr(),
            html.H4("Recommended playlist (with similarity scores)"),
            dash_table.DataTable(id='playlist-table', columns=[
                {"name":"track_idx","id":"track_idx"},
                {"name":"filepath","id":"filepath"},
                {"name":"genre","id":"genre"},
                {"name":"similarity","id":"similarity"}
            ], page_size=10)
        ])
    ])
])

# -------- CALLBACKS --------
@app.callback(
    [Output('audio-preview', 'children'),
     Output('pred-output', 'children'),
     Output('music-graph', 'elements'),
     Output('similar-table', 'data')],
    [Input('upload-audio', 'contents')],
    [State('upload-audio', 'filename')]
)
def analyze_upload(contents, filename):
    if not contents:
        return "", "", [], []
    try:
        y, sr = load_audio_bytes(contents)
        # audio player
        audio_b64 = contents  # same base64 resource
        audio_player = html.Audio(src=audio_b64, controls=True)
        # compute
        pred_genre, probs, emb, mel_img = predict_genre_and_embedding_from_audio(y, sr)
        # similarity
        idxs, sims = top_k_similar(emb, k=8)
        elements = build_cytoscape_elements('query', idxs, sims)
        # table
        rows = []
        for rank,(i,s) in enumerate(zip(idxs, sims)):
            rows.append({
                "track_idx": int(i),
                "filepath": meta.loc[i, 'filepath'],
                "genre": meta.loc[i, 'genre'],
                "similarity": float(s)
            })
        # show spectrogram image
        img_b64 = array_to_base64_png(mel_img)
        pred_div = html.Div([
            html.H4(f"Predicted genre: {pred_genre}"),
            html.Img(src=img_b64, style={'maxWidth':'60%'}),
        ])
        return audio_player, pred_div, elements, rows
    except Exception as e:
        return html.Div(f"Error processing file: {e}"), "", [], []

@app.callback(
    [Output('playlist-table', 'data'), Output('playlist-output', 'children')],
    [Input('gen-playlist-btn', 'n_clicks')],
    [State('all-tracks-table', 'selected_rows')]
)
def generate_playlist(n_clicks, selected_rows):
    if n_clicks <= 0:
        return [], ""
    if not selected_rows or len(selected_rows) < 5:
        return [], html.Div("Select at least 5 favorite tracks.", style={'color':'red'})
    selected_idx = selected_rows
    # compute user profile (mean embedding)
    selected_embeddings = embeddings[selected_idx]
    user_profile = np.mean(selected_embeddings, axis=0).reshape(1, -1)
    sims = cosine_similarity(user_profile, embeddings)[0]
    # exclude selected tracks
    for i in selected_idx:
        sims[i] = -1
    # top 15 recommendations
    top_idx = sims.argsort()[::-1][:15]
    rows = []
    for idx in top_idx:
        rows.append({
            "track_idx": int(idx),
            "filepath": meta.loc[idx, 'filepath'],
            "genre": meta.loc[idx, 'genre'],
            "similarity": float(sims[idx])
        })
    # return table and a small UI summary
    summary = html.Div([
        html.P(f"Generated playlist of {len(rows)} tracks."),
        html.P("You can preview tracks by copying their paths or extend app to stream local files.")
    ])
    return rows, summary

if __name__ == '__main__':
    app.run(debug=True, port=8050)
