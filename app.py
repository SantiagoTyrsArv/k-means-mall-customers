"""
Taller Final Minería de Datos – Segmentación de clientes con K-Means
Aplicación Streamlit: método del codo, silueta, segmentos y clasificación de nuevos clientes.

Ejecutar:  streamlit run app.py
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentos del centro comercial", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="auto")

BASE = Path(__file__).parent
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42
K_MODELO = 5  # K del modelo guardado en el notebook (modelo_kmeans.pkl)

# ---------------------------------------------------------------- Paleta y perfiles
FLOOR, PANEL, INK, MUTED, RULE = "#EDEFEB", "#FFFFFF", "#17201C", "#5B6660", "#D3D9D3"
PIN = "#D7263D"  # nuevo cliente
PERFILES = {
    "Clientes de alto valor": {
        "corto": "Alto valor", "color": "#1E7A55", "suave": "#E1F0E8",
        "objetivo": "Retener y fidelizar",
        "detalle": "Programa VIP con beneficios exclusivos, eventos privados, marcas premium y atención personalizada.",
    },
    "Clientes con potencial": {
        "corto": "Con potencial", "color": "#6D44B8", "suave": "#ECE5F8",
        "objetivo": "Activar y convertir",
        "detalle": "Campañas de marcas de calidad y experiencias premium, no descuentos. "
                   "Investigar por qué gastan poco y enviar ofertas de bienvenida personalizadas.",
    },
    "Clientes promedio": {
        "corto": "Promedio", "color": "#2D5BC4", "suave": "#E4EBFA",
        "objetivo": "Aumentar frecuencia y ticket",
        "detalle": "Programa de puntos escalonado, promociones de temporada, combos y ventas cruzadas.",
    },
    "Clientes con potencial (jóvenes gastadores)": {
        "corto": "Jóvenes gastadores", "color": "#9A6700", "suave": "#F8EED6",
        "objetivo": "Fidelizar a largo plazo",
        "detalle": "Redes sociales e influencers, descuentos para estudiantes, pagos a cuotas "
                   "y gamificación del programa de lealtad.",
    },
    "Clientes de bajo valor": {
        "corto": "Bajo valor", "color": "#B83D0E", "suave": "#F8E6DD",
        "objetivo": "Mantener a bajo costo",
        "detalle": "Ofertas de precio, días de descuento, productos esenciales y comunicación digital automatizada.",
    },
}
ORDEN = list(PERFILES)  # orden de presentación y de las letras A–E
PRIORIDAD = "Clientes con potencial"
LETRAS = "ABCDEFGHIJ"
COLORES_K = ["#2D5BC4", "#1E7A55", "#9A6700", "#6D44B8", "#B83D0E",
             "#0F6B6B", "#A62C70", "#4E5A54", "#64761A", "#2F7FC0"]

# ---------------------------------------------------------------- Estilos
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, h1, h2, h3, [data-testid="stWidgetLabel"],
.hero, .sec, .stApp div {{ font-family: 'Archivo', system-ui, sans-serif !important; font-variant-numeric: tabular-nums; }}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {FLOOR}; color: {INK}; }}
.block-container {{ max-width: 1200px; padding-top: 2.25rem; padding-bottom: 4rem; }}
a:focus-visible, button:focus-visible, input:focus-visible {{ outline: 3px solid #F5C518 !important; outline-offset: 2px; }}

/* Barra lateral: el directorio */
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{ color: #E6ECE8; }}
[data-testid="stSidebar"] hr {{ border-color: #33413A; margin: 18px 0; }}
.dir-title {{ font-stretch: 112%; font-weight: 800; font-size: 22px; line-height: 1.05; margin: 2px 0 4px; letter-spacing: -0.01em; }}
.dir-sub {{ color: #A5B3AB !important; font-size: 13.5px; line-height: 1.5; margin: 0 0 4px; }}
.dir-row {{ display: grid; grid-template-columns: 30px minmax(0,1fr) auto; gap: 10px; align-items: center;
  padding: 9px 0; border-bottom: 1px solid #2A3631; font-size: 14.5px; }}
.dir-row:last-child {{ border-bottom: 0; }}
.dir-count {{ color: #A5B3AB !important; font-size: 13px; }}
.dir-nav a {{ display: block; color: #CBD6D0 !important; text-decoration: none; font-size: 14.5px; padding: 6px 0; }}
.dir-nav a:hover {{ color: #FFFFFF !important; text-decoration: underline; text-underline-offset: 3px; }}

/* Letra de segmento, como en la señalética */
.zona {{ display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 5px;
  color: #FFFFFF !important; font-weight: 800; font-stretch: 112%; font-size: 14px; line-height: 1; flex: none; }}
.zona.big {{ width: 64px; height: 64px; border-radius: 10px; font-size: 38px; }}

/* Encabezado */
.hero {{ font-stretch: 118%; font-weight: 800; font-size: clamp(34px, 5.2vw, 60px); line-height: 0.98;
  letter-spacing: -0.02em; margin: 0 0 14px; color: {INK}; max-width: 16ch; }}
.lead {{ color: {MUTED}; font-size: 17px; line-height: 1.55; max-width: 60ch; margin: 0; }}
.strip {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); margin: 30px 0 6px;
  border-top: 2px solid {INK}; border-bottom: 1px solid {RULE}; }}
.strip > div {{ padding: 14px 18px 16px; border-left: 1px solid {RULE}; }}
.strip > div:first-child {{ border-left: 0; padding-left: 0; }}
.s-label {{ font-size: 13.5px; color: {MUTED}; }}
.s-value {{ font-stretch: 112%; font-weight: 700; font-size: 34px; line-height: 1.1; margin: 4px 0 2px; }}
.s-note {{ font-size: 13px; color: {MUTED}; }}
@media (max-width: 760px) {{
  .strip {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .strip > div:nth-child(3) {{ border-left: 0; padding-left: 0; }}
  .strip > div:nth-child(n+3) {{ border-top: 1px solid {RULE}; }}
}}

/* Secciones */
.sec {{ font-stretch: 115%; font-weight: 750; font-size: 30px; line-height: 1.1; letter-spacing: -0.01em;
  margin: 46px 0 6px; color: {INK}; scroll-margin-top: 70px; }}
.sec-lead {{ color: {MUTED}; font-size: 15.5px; line-height: 1.55; margin: 0 0 16px; max-width: 70ch; }}
.panel {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 10px; }}
div[data-testid="stPlotlyChart"] {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 10px; padding: 10px 6px 2px; }}
div[data-testid="stForm"] {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 10px; padding: 22px; }}

/* Tabla de segmentos */
.tab {{ padding: 6px 20px 14px; }}
.t-row {{ display: grid; grid-template-columns: 26px minmax(0,1fr) 50px 50px 42px 36px; gap: 8px; align-items: center;
  padding: 12px 0; border-bottom: 1px solid #E8ECE8; font-size: 14.5px; }}
.t-head {{ font-size: 12.5px; color: {MUTED}; border-bottom: 1px solid {RULE}; padding: 12px 0 9px; }}
.num {{ text-align: right; }}
.t-note {{ font-size: 12.5px; color: {MUTED}; padding-top: 10px; line-height: 1.5; }}
@media (max-width: 520px) {{ .t-row {{ grid-template-columns: 26px minmax(0,1fr) 50px 50px 42px; }} .t-edad {{ display: none; }} }}

/* Resultado de la clasificación */
.res {{ padding: 24px; display: flex; flex-direction: column; gap: 20px; }}
.res-head {{ display: flex; align-items: center; gap: 16px; }}
.res-name {{ font-stretch: 112%; font-weight: 800; font-size: 28px; line-height: 1.1; }}
.res-sub {{ font-size: 14px; color: {MUTED}; margin-top: 2px; }}
.trio {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE}; }}
.trio > div {{ padding: 12px 14px; border-left: 1px solid {RULE}; }}
.trio > div:first-child {{ border-left: 0; padding-left: 0; }}
.trio b {{ display: block; font-stretch: 110%; font-size: 21px; font-weight: 700; }}
.h4 {{ font-size: 14px; font-weight: 700; margin: 0 0 6px; }}
.bars {{ display: grid; grid-template-columns: 26px minmax(80px, 150px) minmax(0,1fr) 40px; gap: 10px; align-items: center; font-size: 13.5px; }}
.track {{ height: 8px; border-radius: 4px; background: #E8ECE8; overflow: hidden; }}
.track > span {{ display: block; height: 8px; border-radius: 4px; }}
.warn {{ font-size: 13.5px; color: #6B4A00; background: #F8EED6; border-left: 3px solid #9A6700; padding: 10px 12px; border-radius: 4px; }}

/* Estrategias: filas de directorio */
.st-row {{ display: grid; grid-template-columns: 30px minmax(150px, 220px) minmax(0,1fr); gap: 16px; padding: 18px 20px;
  border-bottom: 1px solid #E8ECE8; align-items: start; }}
.st-row:last-child {{ border-bottom: 0; }}
.st-row.prio {{ background: #F6F3FC; box-shadow: inset 4px 0 0 #6D44B8; }}
.st-name {{ font-weight: 700; font-size: 15.5px; }}
.st-meta {{ font-size: 13px; color: {MUTED}; margin-top: 2px; }}
.st-obj {{ font-weight: 700; font-size: 15px; }}
.st-txt {{ font-size: 14.5px; line-height: 1.55; margin: 3px 0 0; color: {INK}; max-width: 68ch; }}
.tag {{ display: inline-block; font-size: 12px; font-weight: 700; color: #4F2F8F; background: #E6DCF7; border-radius: 4px; padding: 2px 7px; margin-left: 8px; vertical-align: 1px; }}
@media (max-width: 640px) {{ .st-row {{ grid-template-columns: 30px minmax(0,1fr); }} .st-row > div:last-child {{ grid-column: 2; }} }}

.stButton > button, div[data-testid="stFormSubmitButton"] > button, div[data-testid="stDownloadButton"] > button {{
  background: {INK}; color: #FFFFFF; border: 0; border-radius: 8px; min-height: 46px; font-weight: 700; }}
div[data-testid="stFormSubmitButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover {{ background: #2C3A33; color: #FFFFFF; }}
.foot {{ font-size: 13px; color: {MUTED}; margin-top: 48px; padding-top: 14px; border-top: 1px solid {RULE}; line-height: 1.6; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Datos y modelo
@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / "Mall_Customers.csv").rename(columns={"Genre": "Gender"})


@st.cache_data
def curvas(X_scaled, k_max=10):
    inercias, siluetas = [], []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan)
    return inercias, siluetas


@st.cache_resource
def entrenar(k):
    """Para K = 5 usa el modelo guardado en el notebook; si no carga (o para otro K), entrena uno nuevo."""
    df = cargar_datos()
    X = df[FEATURES].values
    if k == K_MODELO:
        try:
            # El .pkl es local y lo genera el notebook del taller (Parte 7); no se cargan archivos externos.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                guardado = joblib.load(BASE / "modelo_kmeans.pkl")
            scaler, km = guardado["scaler"], guardado["kmeans"]
            if km.n_clusters == k and list(guardado["features"]) == FEATURES:
                km.labels_ = km.predict(scaler.transform(X))
                return scaler, km, "modelo_kmeans.pkl"
        except Exception:
            pass
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(scaler.transform(X))
    return scaler, km, "entrenado en la app"


def perfil(ingreso, gasto, media_ing, media_gas, tol=0.2):
    """Clasifica un centroide según su ingreso y gasto respecto a la media (tolerancia ±20 %)."""
    ing = "alto" if ingreso > media_ing * (1 + tol) else "bajo" if ingreso < media_ing * (1 - tol) else "medio"
    gas = "alto" if gasto > media_gas * (1 + tol) else "bajo" if gasto < media_gas * (1 - tol) else "medio"
    if ing == "alto" and gas == "alto":
        return "Clientes de alto valor"
    if ing == "alto" and gas == "bajo":
        return "Clientes con potencial"
    if ing == "bajo" and gas == "alto":
        return "Clientes con potencial (jóvenes gastadores)"
    if ing == "bajo" and gas == "bajo":
        return "Clientes de bajo valor"
    return "Clientes promedio"


def html(bloque):
    """Muestra HTML sin sangrías ni líneas vacías (evita que Markdown lo convierta en bloque de código)."""
    st.markdown(" ".join(l.strip() for l in bloque.splitlines() if l.strip()), unsafe_allow_html=True)


def zona(letra, color, big=False):
    return f'<span class="zona{" big" if big else ""}" style="background:{color}">{letra}</span>'


def seccion(ancla, titulo, texto):
    st.markdown(f'<h2 id="{ancla}" class="sec">{titulo}</h2><p class="sec-lead">{texto}</p>', unsafe_allow_html=True)


def estilo_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=14, t=44, b=10), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="Archivo, sans-serif", color=INK, size=13), showlegend=False,
        title_font=dict(size=16, family="Archivo, sans-serif", color=INK), title_x=0.01,
        hoverlabel=dict(font_family="Archivo, sans-serif", bgcolor=INK, font_color="#FFFFFF", bordercolor=INK),
    )
    ejes = dict(gridcolor="#E8ECE8", linecolor="#B7C0BA", tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=12.5), zeroline=False)
    fig.update_xaxes(**ejes)
    fig.update_yaxes(**ejes)
    return fig


GRAF = {"displayModeBar": False}

df = cargar_datos()
X_scaled_all = StandardScaler().fit_transform(df[FEATURES].values)
inercias, siluetas = curvas(X_scaled_all)
k_best = int(np.nanargmax(siluetas)) + 1
media_ing, media_gas = df[FEATURES[0]].mean(), df[FEATURES[1]].mean()

# ---------------------------------------------------------------- Barra lateral (parte 1)
with st.sidebar:
    html("""<div class="dir-title">Directorio de clientes</div>
         <p class="dir-sub">Segmentación K-Means del Mall Customers Dataset. Taller final de Minería de Datos.</p>""")
    st.markdown("---")
    k = st.slider("Número de segmentos (K)", 2, 10, k_best,
                  help=f"El método del codo y el coeficiente de silueta coinciden en K = {k_best}.")

scaler, kmeans, origen = entrenar(k)
df["Cluster"] = kmeans.labels_
centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
perfiles_ok = k == K_MODELO
if perfiles_ok:
    centroides["Perfil"] = [perfil(r[FEATURES[0]], r[FEATURES[1]], media_ing, media_gas) for _, r in centroides.iterrows()]
    perfiles_ok = centroides["Perfil"].nunique() == k  # cada centroide debe tener un perfil distinto
if perfiles_ok:
    centroides["Etiqueta"] = centroides["Perfil"].map(lambda p: PERFILES[p]["corto"])
    centroides["Color"] = centroides["Perfil"].map(lambda p: PERFILES[p]["color"])
    centroides["Letra"] = centroides["Perfil"].map(lambda p: LETRAS[ORDEN.index(p)])
else:
    # Sin perfiles de negocio: letras ordenadas de mayor a menor gasto
    orden_gasto = centroides[FEATURES[1]].sort_values(ascending=False).index
    letra_de = {c: LETRAS[i] for i, c in enumerate(orden_gasto)}
    centroides["Perfil"] = [f"Segmento {letra_de[i]}" for i in centroides.index]
    centroides["Etiqueta"] = centroides["Perfil"]
    centroides["Color"] = [COLORES_K[i % 10] for i in centroides.index]
    centroides["Letra"] = [letra_de[i] for i in centroides.index]
df["Perfil"] = df["Cluster"].map(centroides["Perfil"])
df["Segmento"] = df["Cluster"].map(centroides["Letra"])

resumen = df.groupby("Cluster").agg(n=("CustomerID", "count"), ing=(FEATURES[0], "mean"),
                                    gas=(FEATURES[1], "mean"), edad=("Age", "mean"))
resumen = resumen.join(centroides[["Perfil", "Etiqueta", "Color", "Letra"]]).sort_values("Letra")
sil_k = siluetas[k - 1]
mujeres = int((df["Gender"] == "Female").sum())

# ---------------------------------------------------------------- Barra lateral (parte 2)
with st.sidebar:
    filas = "".join(f'<div class="dir-row">{zona(r.Letra, r.Color)}<span>{r.Etiqueta}</span>'
                    f'<span class="dir-count">{r.n}</span></div>' for r in resumen.itertuples())
    html(f"""<div style="margin-top:14px">{filas}</div>""")
    if not perfiles_ok:
        html(f'<p class="dir-sub" style="margin-top:10px">Los perfiles de negocio se definieron con K = {K_MODELO}.</p>')
    st.markdown("---")
    html("""<nav class="dir-nav">
         <a href="#mapa">Mapa de segmentos</a><a href="#clasificar">Clasificar un cliente</a>
         <a href="#elegir-k">Cómo se eligió K</a><a href="#estrategias">Estrategias</a>
         <a href="#exploracion">Exploración de los datos</a></nav>""")
    st.markdown("---")
    html("""<p class="dir-sub">Variables del modelo: ingreso anual (k$) y puntaje de gasto (1–100),
         estandarizadas con StandardScaler. Edad y género solo describen los segmentos.</p>""")

# ---------------------------------------------------------------- Encabezado
reduccion = (1 - inercias[k - 1] / inercias[k - 2]) * 100
html(f"""
<h1 class="hero">¿Quién compra en el centro comercial?</h1>
<p class="lead">{len(df)} clientes agrupados según cuánto ganan y cuánto gastan. Elige el número de segmentos en el
directorio de la barra lateral, revisa el mapa y clasifica a un cliente nuevo para ver qué estrategia le corresponde.</p>
<div class="strip">
  <div><div class="s-label">Clientes</div><div class="s-value">{len(df)}</div>
    <div class="s-note">{mujeres} mujeres, {len(df) - mujeres} hombres</div></div>
  <div><div class="s-label">Segmentos (K)</div><div class="s-value">{k}</div>
    <div class="s-note">{"Óptimo según codo y silueta" if k == k_best else f"El óptimo es K = {k_best}"}</div></div>
  <div><div class="s-label">Coeficiente de silueta</div><div class="s-value">{sil_k:.3f}</div>
    <div class="s-note">{"Grupos bien separados" if sil_k > 0.5 else "Separación moderada"}</div></div>
  <div><div class="s-label">Inercia (WCSS)</div><div class="s-value">{kmeans.inertia_:.1f}</div>
    <div class="s-note">{reduccion:.0f}&nbsp;% menos que con K&nbsp;=&nbsp;{k - 1}</div></div>
</div>
""")


# ---------------------------------------------------------------- Mapa de segmentos
def figura_segmentos(nuevo=None):
    fig = go.Figure()
    for i, r in centroides.iterrows():
        sub = df[df["Cluster"] == i]
        fig.add_trace(go.Scatter(
            x=sub[FEATURES[0]], y=sub[FEATURES[1]], mode="markers", name=r["Etiqueta"],
            marker=dict(size=9, color=r["Color"], opacity=0.78, line=dict(width=0)),
            customdata=np.stack([sub["CustomerID"], sub["Age"]], axis=-1),
            hovertemplate=f"<b>{r['Letra']}. {r['Etiqueta']}</b><br>Cliente %{{customdata[0]}}, %{{customdata[1]}} años"
                          "<br>Ingreso %{x} k$, gasto %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=centroides[FEATURES[0]], y=centroides[FEATURES[1]], mode="markers+text", text=centroides["Letra"],
        textfont=dict(color="#FFFFFF", size=14, family="Archivo, sans-serif"), textposition="middle center",
        marker=dict(size=28, symbol="square", color=centroides["Color"], line=dict(color="#FFFFFF", width=2.5)),
        customdata=centroides["Etiqueta"],
        hovertemplate="<b>Centroide %{text}: %{customdata}</b><br>Ingreso %{x:.1f} k$, gasto %{y:.1f}<extra></extra>"))
    if nuevo is not None:
        fig.add_trace(go.Scatter(
            x=[nuevo[0]], y=[nuevo[1]], mode="markers",
            marker=dict(size=18, color=PIN, line=dict(color="#FFFFFF", width=3)),
            hovertemplate="Nuevo cliente<br>Ingreso %{x} k$, gasto %{y}<extra></extra>"))
        fig.add_annotation(x=nuevo[0], y=nuevo[1], text="Nuevo cliente", showarrow=True, arrowhead=0, arrowwidth=1.5,
                           arrowcolor=PIN, ax=-58 if nuevo[0] > media_ing else 58, ay=-34 if nuevo[1] < 75 else 34,
                           font=dict(color="#FFFFFF", size=13, family="Archivo, sans-serif"),
                           bgcolor=PIN, borderpad=4)
    fig.add_vline(x=media_ing, line=dict(color="#B7C0BA", dash="dot", width=1))
    fig.add_hline(y=media_gas, line=dict(color="#B7C0BA", dash="dot", width=1))
    fig.update_layout(xaxis_title="Ingreso anual (k$)", yaxis_title="Puntaje de gasto (1–100)")
    fig.update_yaxes(range=[-4, 106])
    return estilo_fig(fig, 480)


seccion("mapa", "Mapa de segmentos",
        "Cada punto es un cliente y cada cuadro con letra es el centro de su segmento. Las líneas punteadas marcan "
        "el ingreso y el gasto promedio. Pasa el cursor sobre un punto para ver el detalle.")
c1, c2 = st.columns([1.5, 1], gap="medium")
with c1:
    zona_grafico = st.empty()
with c2:
    filas = "".join(
        f'<div class="t-row">{zona(r.Letra, r.Color)}<span>{r.Etiqueta}</span><span class="num">{r.n}</span>'
        f'<span class="num">{r.ing:.1f}</span><span class="num">{r.gas:.1f}</span>'
        f'<span class="num t-edad">{r.edad:.0f}</span></div>' for r in resumen.itertuples())
    html(f"""<div class="panel tab">
      <div class="t-row t-head"><span></span><span>Segmento</span><span class="num">Clientes</span>
        <span class="num">Ingreso</span><span class="num">Gasto</span><span class="num t-edad">Edad</span></div>
      {filas}
      <div class="t-note">Promedios de cada segmento. Ingreso en miles de dólares al año; gasto de 1 a 100;
        edad en años.</div>
    </div>""")

# ---------------------------------------------------------------- Clasificar
seccion("clasificar", "Clasificar un cliente nuevo",
        "El modelo asigna al cliente el segmento cuyo centro está más cerca, usando ingreso y gasto. "
        "La edad y el género se registran como contexto.")
c1, c2 = st.columns([1, 1.35], gap="medium")
with c1:
    with st.form("nuevo_cliente"):
        f1, f2 = st.columns(2)
        ingreso = f1.number_input("Ingreso anual (k$)", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
        edad = f2.number_input("Edad", min_value=18, max_value=100, value=38)
        gasto = st.slider("Puntaje de gasto (1–100)", 1, 100, 15)
        genero = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
        st.form_submit_button("Clasificar cliente", width="stretch")

x_new = scaler.transform(np.array([[ingreso, gasto]]))
cluster = int(kmeans.predict(x_new)[0])
dist = np.linalg.norm(kmeans.cluster_centers_ - x_new, axis=1)
fila = centroides.loc[cluster]
info = PERFILES.get(fila["Perfil"], {"corto": fila["Etiqueta"], "color": fila["Color"], "objetivo": None,
                                      "detalle": f"Las estrategias se definieron para los segmentos de K = {K_MODELO}."})
seg = resumen.loc[cluster]
dmax = max(dist.max(), 1e-9)
barras = "".join(
    f'{zona(centroides.loc[i, "Letra"], centroides.loc[i, "Color"])}<span>{centroides.loc[i, "Etiqueta"]}</span>'
    f'<span class="track"><span style="width:{max(4, (1 - dist[i] / (dmax * 1.15)) * 100):.0f}%;'
    f'background:{centroides.loc[i, "Color"]}"></span></span><span class="num">{dist[i]:.2f}</span>'
    for i in np.argsort(dist))
fuera = []
for col, valor, unidad in [(FEATURES[0], ingreso, " k$"), ("Age", edad, " años")]:
    lo, hi = df[col].min(), df[col].max()
    if not lo <= valor <= hi:
        fuera.append(f"{'El ingreso' if col == FEATURES[0] else 'La edad'} está fuera del rango de los datos "
                     f"({lo}–{hi}{unidad})")
aviso = (f'<div class="warn">{"; ".join(fuera)}. La clasificación es una extrapolación.</div>' if fuera else "")
objetivo = f'<div class="h4">{info["objetivo"]}</div>' if info.get("objetivo") else '<div class="h4">Estrategia</div>'
with c2:
    html(f"""<div class="panel res">
      <div class="res-head">{zona(fila["Letra"], fila["Color"], big=True)}
        <div><div class="res-sub">Segmento asignado</div><div class="res-name">{fila["Perfil"]}</div>
          <div class="res-sub">{genero.lower()}, {edad} años, ingreso {ingreso:.0f} k$, gasto {gasto}</div></div>
      </div>
      <div class="trio">
        <div><span class="s-note">Clientes en el segmento</span><b>{seg.n}</b></div>
        <div><span class="s-note">Ingreso medio</span><b>{seg.ing:.1f} k$</b></div>
        <div><span class="s-note">Gasto medio</span><b>{seg.gas:.1f}</b></div>
      </div>
      <div>{objetivo}<p style="margin:0;font-size:15px;line-height:1.55">{info["detalle"]}</p></div>
      {aviso}
      <div><div class="h4" style="margin-bottom:10px">Distancia a cada centro</div>
        <div class="bars">{barras}</div>
        <div class="s-note" style="margin-top:10px">En unidades estandarizadas: menor distancia, más parecido.</div></div>
    </div>""")

zona_grafico.plotly_chart(figura_segmentos((ingreso, gasto)), width="stretch", config=GRAF)

# ---------------------------------------------------------------- Elegir K
red_45 = (1 - inercias[4] / inercias[3]) * 100
red_56 = (1 - inercias[5] / inercias[4]) * 100
seccion("elegir-k", "Cómo se eligió K",
        f"La inercia siempre baja al agregar segmentos, así que se busca el codo: de K = 4 a 5 cae&nbsp;{red_45:.0f}&nbsp;%, "
        f"y de 5 a 6 solo {red_56:.0f}&nbsp;%. La silueta, que mide qué tan separados están los grupos, alcanza su "
        f"máximo ({np.nanmax(siluetas):.3f}) en K = {k_best}. Los dos métodos coinciden.")
c1, c2 = st.columns(2)
ks = list(range(1, 11))


def curva(x, y, titulo, ytitulo, color, fmt, rango):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=color, width=2.5),
                               marker=dict(size=7), hovertemplate=f"K = %{{x}}<br>{ytitulo} = %{{y:{fmt}}}<extra></extra>"))
    fig.add_vline(x=k_best, line=dict(color="#B7C0BA", dash="dot", width=1))
    y_k = y[x.index(k)]
    fig.add_trace(go.Scatter(x=[k], y=[y_k], mode="markers+text", text=[f"K = {k}: {y_k:{fmt}}"],
                             textposition="top right", textfont=dict(color=INK, size=13),
                             marker=dict(size=14, color=PANEL, line=dict(color=INK, width=3)), hoverinfo="skip"))
    fig.update_layout(title=titulo, xaxis_title="Número de segmentos (K)", yaxis_title=ytitulo)
    fig.update_yaxes(range=rango)
    fig.update_xaxes(dtick=1)
    return estilo_fig(fig, 330)


with c1:
    st.plotly_chart(curva(ks, inercias, "Método del codo", "Inercia (WCSS)", "#2D5BC4", ".1f",
                          [0, max(inercias) * 1.12]), width="stretch", config=GRAF)
with c2:
    st.plotly_chart(curva(ks[1:], siluetas[1:], "Coeficiente de silueta", "Silueta", "#1E7A55", ".3f",
                          [min(siluetas[1:]) - 0.03, max(siluetas[1:]) + 0.06]), width="stretch", config=GRAF)

# ---------------------------------------------------------------- Estrategias
if perfiles_ok:
    seccion("estrategias", "Estrategias por segmento",
            "La mayor oportunidad está en los clientes de ingreso alto y gasto bajo: tienen capacidad de compra "
            "que hoy no gastan en el centro comercial.")
    filas = ""
    for r in resumen.itertuples():
        d = PERFILES[r.Perfil]
        prio = r.Perfil == PRIORIDAD
        filas += (f'<div class="st-row{" prio" if prio else ""}">{zona(r.Letra, r.Color)}'
                  f'<div><div class="st-name">{d["corto"]}</div>'
                  f'<div class="st-meta">{r.n} clientes ({r.n / len(df) * 100:.1f}&nbsp;%)</div></div>'
                  f'<div><div class="st-obj">{d["objetivo"]}{"<span class=tag>Mayor oportunidad</span>" if prio else ""}</div>'
                  f'<p class="st-txt">{d["detalle"]}</p></div></div>')
    html(f'<div class="panel">{filas}</div>')
else:
    seccion("estrategias", "Estrategias por segmento",
            f"Las estrategias se definieron para los cinco segmentos de K = {K_MODELO}. "
            f"Vuelve a K = {K_MODELO} en el directorio para verlas.")

# ---------------------------------------------------------------- Exploración (Parte 1)
seccion("exploracion", "Exploración de los datos",
        "200 clientes, sin valores nulos ni filas duplicadas. Solo el ingreso tiene dos valores atípicos "
        "(137 k$), que se conservan porque son plausibles. Ingreso y gasto no están correlacionados.")
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
nombres = {"Age": "Edad (años)", "Annual Income (k$)": "Ingreso anual (k$)", "Spending Score (1-100)": "Puntaje de gasto"}
cols = st.columns(3)
for col_st, col in zip(cols, num_cols):
    fig = go.Figure(go.Histogram(x=df[col], nbinsx=16, marker=dict(color="#4E5A54", line=dict(color=PANEL, width=1)),
                                 hovertemplate="%{x}<br>%{y} clientes<extra></extra>"))
    fig.update_layout(title=nombres[col], bargap=0.04, yaxis_title="Clientes")
    col_st.plotly_chart(estilo_fig(fig, 250), width="stretch", config=GRAF)

c1, c2 = st.columns([3, 2], gap="medium")
with c1:
    desc = df[num_cols].describe().T.round(1).rename(index=nombres)
    desc.columns = ["Clientes", "Media", "Desv.", "Mín.", "25 %", "Mediana", "75 %", "Máx."]
    st.dataframe(desc, width="stretch")
with c2:
    corr = df[num_cols].corr().round(2)
    etiquetas = ["Edad", "Ingreso", "Gasto"]
    fig = go.Figure(go.Heatmap(z=corr.values, x=etiquetas, y=etiquetas, zmin=-1, zmax=1, text=corr.values,
                               texttemplate="%{text:.2f}", colorscale=[[0, "#B83D0E"], [0.5, "#FFFFFF"], [1, "#2D5BC4"]],
                               showscale=False, hovertemplate="%{y} y %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Correlación de Pearson")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo_fig(fig, 250), width="stretch", config=GRAF)

with st.expander("Ver los 200 clientes con su segmento"):
    tabla = df.rename(columns={"Gender": "Género", "Age": "Edad"})
    st.dataframe(tabla, hide_index=True, height=380, width="stretch")
    st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8"), "clientes_segmentados.csv",
                       "text/csv")

html(f"""<div class="foot">Modelo K-Means (k-means++, n_init = 10, random_state = {RANDOM_STATE}) sobre ingreso y gasto
estandarizados con StandardScaler. Modelo para K = {k}: {origen}. Datos: Mall Customers Dataset (Kaggle).</div>""")
