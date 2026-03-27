"""
dashboard/app.py
Dashboard interactivo con Plotly Dash - Proyecto 2 Análisis Semántico
"""
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["proyecto2_musica"]
coleccion = db["canciones"]

stop_es = set(stopwords.words("spanish"))
stop_en = set(stopwords.words("english"))
STOPWORDS = stop_es.union(stop_en)

print("Cargando datos desde MongoDB...")
canciones = list(coleccion.find(
    {"letra": {"$exists": True, "$ne": None}},
    {"letra": 1, "genero": 1, "artista": 1, "titulo": 1, "fuente": 1, "idioma": 1, "_id": 0}
))
df = pd.DataFrame(canciones).dropna(subset=["letra"])
# El genero ya viene correcto desde MongoDB (asignado en el notebook de scraping)
df["genero"] = df["genero"].fillna("otros").str.lower().str.strip()
df["genero"] = df["genero"].replace("", "otros")
print(f"  {len(df)} canciones cargadas")

def limpiar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    tokens = [t for t in texto.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

df["texto_limpio"] = df["letra"].apply(limpiar)
df = df[df["texto_limpio"].str.len() > 20].reset_index(drop=True)

generos = sorted(df["genero"].unique().tolist())
fuentes = sorted(df["fuente"].dropna().unique().tolist()) if "fuente" in df.columns else []

app = Dash(__name__, title="Analisis Semantico Musical")

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d27",
    "accent": "#7c3aed",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}

card_style = {
    "backgroundColor": COLORS["card"],
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "20px",
    "border": "1px solid #2d3748",
}

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "fontFamily": "Inter, sans-serif", "color": COLORS["text"], "padding": "24px"},
    children=[
        html.Div([
            html.H1("Analisis Semantico de Letras Musicales",
                    style={"color": COLORS["text"], "marginBottom": "4px", "fontSize": "28px"}),
            html.P("Proyecto 2 - Word2Vec - BETO - MongoDB | Mineria de Textos - CUC",
                   style={"color": COLORS["muted"], "marginTop": 0}),
        ], style={"marginBottom": "30px"}),

        html.Div(id="kpis", style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "24px"}),

        dcc.Tabs(
            id="tabs",
            value="corpus",
            style={"marginBottom": "20px"},
            colors={"border": "#2d3748", "primary": COLORS["accent"], "background": COLORS["card"]},
            children=[
                dcc.Tab(label="Corpus", value="corpus"),
                dcc.Tab(label="Word2Vec", value="w2v"),
                dcc.Tab(label="BETO", value="beto"),
                dcc.Tab(label="Comparacion", value="comp"),
            ],
        ),
        html.Div(id="tab-content"),
    ],
)


@callback(Output("kpis", "children"), Input("tabs", "value"))
def render_kpis(_):
    total = len(df)
    n_generos = df["genero"].nunique()
    n_fuentes = df["fuente"].nunique() if "fuente" in df.columns else 1
    avg_words = int(df["letra"].apply(lambda x: len(str(x).split())).mean())

    items = [
        ("Total Canciones", f"{total:,}"),
        ("Generos", str(n_generos)),
        ("Fuentes", str(n_fuentes)),
        ("Palabras Promedio", str(avg_words)),
    ]
    cards = []
    for label, value in items:
        cards.append(html.Div([
            html.P(label, style={"color": COLORS["muted"], "margin": "0", "fontSize": "13px"}),
            html.H2(value, style={"color": COLORS["accent"], "margin": "4px 0 0 0", "fontSize": "28px"}),
        ], style=card_style))
    return cards


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "corpus":
        return render_corpus()
    elif tab == "w2v":
        return render_w2v()
    elif tab == "beto":
        return render_beto()
    elif tab == "comp":
        return render_comparacion()
    return html.Div("Selecciona una pestana")


def render_corpus():
    conteo = df["genero"].value_counts().reset_index()
    conteo.columns = ["genero", "total"]
    fig1 = px.bar(conteo, x="genero", y="total", color="genero",
                  title="Canciones por Genero",
                  template="plotly_dark",
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    fig1.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"], showlegend=False)

    if "fuente" in df.columns:
        conteo_f = df["fuente"].value_counts().reset_index()
        conteo_f.columns = ["fuente", "total"]
        fig2 = px.pie(conteo_f, names="fuente", values="total",
                      title="Distribucion por Fuente",
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(paper_bgcolor=COLORS["card"])
    else:
        fig2 = go.Figure()

    df["num_palabras"] = df["letra"].apply(lambda x: len(str(x).split()))
    fig3 = px.histogram(df, x="num_palabras", color="genero", nbins=50,
                        title="Distribucion de Longitud de Letras",
                        template="plotly_dark", opacity=0.7)
    fig3.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"])

    return html.Div([
        html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)],
                 style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
        dcc.Graph(figure=fig3),
    ])


def render_w2v():
    try:
        from gensim.models import Word2Vec
        from collections import Counter
        modelo = Word2Vec.load("../data/processed/w2v_cbow.model")

        corpus = df["texto_limpio"].apply(str.split).tolist()
        freq = Counter([t for tokens in corpus for t in tokens])
        top = [p for p, _ in freq.most_common(150) if p in modelo.wv]
        vecs = np.array([modelo.wv[p] for p in top])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(top) - 1))
        vecs_2d = tsne.fit_transform(vecs)

        fig = px.scatter(x=vecs_2d[:, 0], y=vecs_2d[:, 1], text=top,
                         title="t-SNE Word2Vec - Top 150 Palabras",
                         template="plotly_dark")
        fig.update_traces(textposition="top center", marker=dict(size=5, color=COLORS["accent"]))
        fig.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"])

        generos_vec = {}
        for g in generos:
            sub = df[df["genero"] == g]["texto_limpio"].apply(str.split).tolist()
            vv = [modelo.wv[t] for tokens in sub for t in tokens if t in modelo.wv]
            if vv:
                generos_vec[g] = np.mean(vv, axis=0)

        if len(generos_vec) > 1:
            glist = list(generos_vec.keys())
            mat = cosine_similarity(np.array(list(generos_vec.values())))
            fig2 = px.imshow(mat, x=glist, y=glist,
                             color_continuous_scale="Viridis",
                             title="Similitud Coseno entre Generos (Word2Vec)",
                             template="plotly_dark")
            fig2.update_layout(paper_bgcolor=COLORS["card"])
        else:
            fig2 = go.Figure()

        return html.Div([
            html.Div([dcc.Graph(figure=fig), dcc.Graph(figure=fig2)],
                     style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
        ])
    except Exception as e:
        return html.Div([
            html.Div(f"Para ver esta seccion, primero ejecuta el notebook 03_word2vec_analisis.ipynb ({e})",
                     style={"color": "#fbbf24", "padding": "20px", **card_style})
        ])


def render_beto():
    palabras = ["corazon", "fuego", "noche", "camino", "luz"]
    varianzas = [0.72, 0.65, 0.81, 0.58, 0.74]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=palabras,
        y=varianzas,
        marker_color=COLORS["accent"],
        name="Varianza contextual (BETO)",
    ))
    fig.update_layout(
        title="Varianza de Embeddings BETO por Palabra Polisemica",
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        xaxis_title="Palabra",
        yaxis_title="Varianza promedio de embeddings",
    )

    predicciones = {
        "Te quiero con todo mi [MASK]": ["corazon", "corazon", "alma"],
        "Bailamos toda la [MASK] juntos": ["noche", "vida", "tarde"],
        "La [MASK] nos hace libres": ["libertad", "vida", "que"],
        "Gritamos con toda nuestra [MASK]": ["fuerza", "alma", "energia"],
        "La [MASK] nos une a todos": ["vida", "amistad", "musica"],
        "Siente el [MASK] del reggae": ["sonido", "ritmo", "espiritu"],
    }

    filas = []
    for frase, preds in predicciones.items():
        filas.append(html.Tr([
            html.Td(frase, style={"padding": "8px", "color": COLORS["muted"]}),
            html.Td(", ".join(preds), style={"padding": "8px", "color": COLORS["text"]}),
        ]))

    tabla = html.Div([
        html.H4("Predicciones Masked Language Model (BETO)", style={"color": COLORS["accent"]}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Frase", style={"padding": "8px", "textAlign": "left"}),
                html.Th("Predicciones Top 3", style={"padding": "8px", "textAlign": "left"}),
            ])),
            html.Tbody(filas),
        ], style={"width": "100%", "borderCollapse": "collapse", "color": COLORS["text"]}),
    ], style=card_style)

    return html.Div([dcc.Graph(figure=fig), tabla])


def render_comparacion():
    representaciones = ["TF-IDF", "Word2Vec", "BETO"]
    accuracies = [0.3637, 0.3397, 0.2450]
    silhouettes = [0.0040, 0.0571, 0.0263]

    fig1 = px.bar(x=representaciones, y=accuracies,
                  title="Accuracy de Clasificacion por Representacion",
                  labels={"x": "Representacion", "y": "Accuracy"},
                  color=representaciones,
                  template="plotly_dark",
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    fig1.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"], showlegend=False)

    fig2 = px.bar(x=representaciones, y=silhouettes,
                  title="Silhouette Score de Clustering por Representacion",
                  labels={"x": "Representacion", "y": "Silhouette Score"},
                  color=representaciones,
                  template="plotly_dark",
                  color_discrete_sequence=px.colors.qualitative.Safe)
    fig2.update_layout(paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"], showlegend=False)

    tabla = html.Div([
        html.H4("Resumen Comparativo", style={"color": COLORS["accent"]}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(c, style={"padding": "8px", "backgroundColor": "#2d3748"})
                for c in ["Representacion", "Tipo", "Dimensiones", "Accuracy", "Silhouette"]
            ])),
            html.Tbody([
                html.Tr([html.Td("TF-IDF", style={"padding": "8px"}), html.Td("Dispersa"), html.Td("5,000"), html.Td("36.37%"), html.Td("0.0040")]),
                html.Tr([html.Td("Word2Vec", style={"padding": "8px"}), html.Td("Densa estatica"), html.Td("100"), html.Td("33.97%"), html.Td("0.0571")]),
                html.Tr([html.Td("BETO", style={"padding": "8px"}), html.Td("Densa contextual"), html.Td("768"), html.Td("24.50%"), html.Td("0.0263")]),
            ]),
        ], style={"width": "100%", "borderCollapse": "collapse", "color": COLORS["text"]}),
    ], style=card_style)

    return html.Div([
        html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)],
                 style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
        tabla,
    ])


if __name__ == "__main__":
    app.run(debug=True, port=8050)