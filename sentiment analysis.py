import streamlit as st
import warnings
import re
import html
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

warnings.filterwarnings('ignore')

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Media Sentiment Intelligence | US-Israel-Iran War",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,179,237,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #f0f4ff;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6);
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #a0aec0;
    margin: 0;
    font-weight: 300;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,179,237,0.2);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.metric-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #63b3ed;
    margin: 0;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #f7fafc;
    border-left: 4px solid #667eea;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #f0fff4, #e6fffa);
    border-left: 4px solid #38a169;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.4rem;
    margin: 0.75rem 0;
    font-size: 0.92rem;
    color: #2d3748;
}
.insight-box.warn {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border-left-color: #e53e3e;
}
.insight-box.info {
    background: linear-gradient(135deg, #ebf8ff, #bee3f8);
    border-left-color: #3182ce;
}

/* Tab styling */
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 100%);
}
[data-testid="stSidebar"] * {
    color: #f7fafc !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #63b3ed !important;
    font-family: 'DM Serif Display', serif !important;
}

/* Status pills */
.pill-neg { background:#fee2e2; color:#991b1b; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:600; }
.pill-pos { background:#dcfce7; color:#166534; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:600; }
.pill-neu { background:#f3f4f6; color:#374151; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─── NLTK Downloads ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    for pkg in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'punkt_tab']:
        nltk.download(pkg, quiet=True)

download_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─── NLP Utilities ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_nlp_tools():
    base_sw = set(stopwords.words('english'))
    negation = {'no','not','nor','never',"don't","didn't","doesn't","won't","wouldn't",
                "can't","couldn't","shouldn't","isn't","aren't","wasn't","weren't",
                "hasn't","haven't","hadn't"}
    base_sw -= negation
    custom_sw = {'said','say','says','would','could','also','mr','mrs','ms','dr',
                 'bbc','cnn','reuters','ap','afp','rt','copyright','rights','reserved',
                 'external','sites','site','linking','approach','read','watch','video',
                 'live','breaking','news','http','https','www','com','amp','u','s'}
    stop_words = base_sw | custom_sw
    lemmatizer = WordNetLemmatizer()
    analyzer   = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, analyzer

stop_words, lemmatizer, analyzer = build_nlp_tools()


def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'
    elif tag.startswith('V'): return 'v'
    elif tag.startswith('N'): return 'n'
    elif tag.startswith('R'): return 'r'
    return 'n'


def remove_boilerplate(text):
    for p in [r'copyright\s+\d{4}.*', r'all rights reserved.*',
              r'the bbc is not responsible.*', r'read about our approach.*']:
        text = re.sub(p, ' ', text, flags=re.IGNORECASE)
    return text


def preprocess(text, is_comment=False):
    if pd.isna(text): return ''
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\bU\.S\.A\.\b','usa',text,flags=re.IGNORECASE)
    text = re.sub(r'\bU\.S\.\b',  'us', text,flags=re.IGNORECASE)
    text = re.sub(r'\bU\.K\.\b',  'uk', text,flags=re.IGNORECASE)
    text = re.sub(r'\bU\.N\.\b',  'un', text,flags=re.IGNORECASE)
    text = re.sub(r'\bI\.R\.G\.C\.\b','irgc',text,flags=re.IGNORECASE)
    text = text.replace('â\x80\x99',"'").replace('â\x80\x93',' ')
    text = text.replace('â\x80\x94',' ').replace('\x92',"'").replace('\xa0',' ')
    if not is_comment: text = remove_boilerplate(text)
    text = re.sub(r'http\S+|www\S+',' ',text)
    if is_comment:
        text = re.sub(r'@\w+',' ',text)
        text = re.sub(r'#','',text)
        text = re.sub(r'[^\x00-\x7F]+',' ',text)
        text = re.sub(r'(.)\1{2,}',r'\1\1',text)
    text = text.lower()
    text = re.sub(r"[^a-z'\s]",' ',text)
    text = re.sub(r'\s+',' ',text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t)>1 or t in {'i','a'}]
    tagged = pos_tag(tokens)
    clean_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(t))
                    for w,t in tagged
                    if lemmatizer.lemmatize(w,get_wordnet_pos(t)) not in stop_words]
    return ' '.join(clean_tokens)


def score(text):
    return analyzer.polarity_scores(str(text))['compound']

def label(s):
    if s >= 0.05:  return 'Positive'
    elif s <= -0.05: return 'Negative'
    else: return 'Neutral'


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(articles_file, comments_file):
    articles_df = pd.read_csv(articles_file, encoding='latin1')
    articles_df = articles_df[['SOURCE','TEXT']].dropna(subset=['TEXT']).reset_index(drop=True)
    articles_df.columns = ['source','text']
    articles_df['source'] = articles_df['source'].replace({'aljazeera':'Al Jazeera'})
    articles_df['data_type'] = 'News Article'

    comments_df = pd.read_csv(comments_file, encoding='latin1')
    if 'comment_text' in comments_df.columns:
        comments_df = comments_df.rename(columns={'comment_text':'text'})
    comments_df = comments_df[['source','text']].dropna(subset=['text']).reset_index(drop=True)
    comments_df['data_type'] = 'Comment'
    return articles_df, comments_df


@st.cache_data(show_spinner=False)
def preprocess_data(_articles_df, _comments_df):
    art = _articles_df.copy()
    com = _comments_df.copy()
    art['clean_text'] = art['text'].apply(lambda x: preprocess(x, is_comment=False))
    com['clean_text'] = com['text'].apply(lambda x: preprocess(x, is_comment=True))
    art = art[art['clean_text'].str.strip()!=''].reset_index(drop=True)
    com = com[com['clean_text'].str.strip()!=''].reset_index(drop=True)
    art['article_id'] = art.index
    return art, com


@st.cache_data(show_spinner=False)
def score_articles(_articles_df):
    df = _articles_df.copy()
    df['sentiment_score'] = df['clean_text'].apply(score)
    df['sentiment_label'] = df['sentiment_score'].apply(label)
    return df


@st.cache_data(show_spinner=False)
def score_comments(_comments_df):
    df = _comments_df.copy()
    df['sentiment_score'] = df['clean_text'].apply(score)
    df['sentiment_label'] = df['sentiment_score'].apply(label)
    return df


@st.cache_data(show_spinner=False)
def build_sentence_df(_articles_df):
    rows = []
    for _, row in _articles_df.iterrows():
        for sent in sent_tokenize(str(row['text'])):
            sent = sent.strip()
            if len(sent.split()) >= 4:
                s = score(sent)
                rows.append({'article_id':row['article_id'],'source':row['source'],
                             'sentence':sent,'score':s,'label':label(s)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_combined_df(_articles_df, _comments_df):
    art_rows = []
    for _, row in _articles_df.iterrows():
        for sent in sent_tokenize(str(row['text'])):
            sent = sent.strip()
            if len(sent.split()) >= 4:
                s = score(sent)
                art_rows.append({'source':row['source'],'data_type':'News Article',
                                 'text':sent,'score':s,'label':label(s)})
    com_rows = []
    for _, row in _comments_df.iterrows():
        s = score(row['text'])
        com_rows.append({'source':row['source'],'data_type':'Comment',
                         'text':row['text'],'score':s,'label':label(s)})
    return pd.DataFrame(art_rows), pd.DataFrame(com_rows)


@st.cache_data(show_spinner=False)
def run_lda(_articles_df):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    texts = _articles_df['clean_text'].dropna().tolist()
    vectorizer = CountVectorizer(max_df=0.90, min_df=3, max_features=1000)
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=7, random_state=42, max_iter=20)
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topic_labels = [
        'Military Strikes','Diplomatic Negotiations','Economic/Energy Impact',
        'Regional Geopolitics','Civilian Impact','Iran Leadership/IRGC','International Response'
    ]
    topic_data = []
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        lbl = topic_labels[idx] if idx < len(topic_labels) else f'Topic {idx+1}'
        topic_data.append({'Topic':lbl,'Top Keywords':', '.join(top_words)})
    topic_assignments = lda.transform(dtm).argmax(axis=1)
    return pd.DataFrame(topic_data), topic_assignments, topic_labels, lda, vectorizer, feature_names


# ─── Angle & Entity Helpers ───────────────────────────────────────────────────
ANGLES = {
    'Military Operations': ['missile','drone','airstrike','strike','bomb','military','attack','weapon',
                             'defense','navy','fighter','jet','irgc','centcom','operation','warship',
                             'carrier','troops','artillery','ballistic','launch','intercept'],
    'Geopolitical Tensions': ['sanction','ally','alliance','tension','region','gulf','ceasefire',
                               'diplomacy','china','russia','europe','nato','un','sovereignty',
                               'international','escalat','neutral','mediat'],
    'Economic Impact': ['oil','energy','price','market','barrel','gas','hormuz','inflation',
                        'economy','supply','trade','recession','lng','fuel','cost','crude',
                        'shipping','tanker','export','stock','financial'],
    'Media Narratives & Propaganda': ['propaganda','misinformation','bias','narrative','fake',
                                       'disinformation','media','distrust','mislead','manipulate',
                                       'social media','youtube','facebook','twitter','report','coverage','frame'],
    'Support for the War': ['support','oppose','protest','condemn','approve','justify','coalition',
                             'against','favor','opposition','opinion','public','poll','ally','stance','position','response']
}

FRAMES = {
    'Conflict Frame':     ['attack','missile','war','strike','bomb','military','kill','destroy'],
    'Humanitarian Frame': ['civilian','death','injury','hospital','refugee','aid','victim','children'],
    'Diplomacy Frame':    ['talk','negotiation','ceasefire','agreement','peace','deal','mediat'],
    'Legal Frame':        ['crime','law','court','aggression','sanction','illegal','violation']
}

ENTITIES = ['iran','israel','us','trump']

COLOR_SENT = {'Negative':'#E74C3C','Neutral':'#95A5A6','Positive':'#2ECC71'}
COLOR_TYPE = {'News Article':'#3498DB','Comment':'#E67E22'}


# ─── Plotting helpers ─────────────────────────────────────────────────────────
def plotly_defaults(fig):
    fig.update_layout(
        plot_bgcolor='rgba(30,30,50,0.0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans, sans-serif', size=11, color='#e2e8f0'),
        title_font=dict(color='#f7fafc'),
        legend=dict(
            font=dict(color='#e2e8f0'),
            bgcolor='rgba(15,12,41,0.6)',
            bordercolor='rgba(255,255,255,0.15)',
            borderwidth=1
        ),
        xaxis=dict(
            color='#cbd5e0',
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.15)',
            tickfont=dict(color='#cbd5e0'),
            title_font=dict(color='#a0aec0')
        ),
        yaxis=dict(
            color='#cbd5e0',
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.15)',
            tickfont=dict(color='#cbd5e0'),
            title_font=dict(color='#a0aec0')
        ),
        margin=dict(t=60, b=40, l=40, r=20)
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📰 Sentiment Intelligence")
    st.markdown("**US/Israel–Iran War Coverage**")
    st.divider()

    st.markdown("### 📂 Upload Your Data")
    articles_file = st.file_uploader("Articles CSV", type=['csv'], key='art')
    comments_file = st.file_uploader("Comments CSV", type=['csv'], key='com')

    st.divider()
    st.markdown("### 🔬 About This App")
    st.markdown("""
    This dashboard provides a complete NLP sentiment analysis of war coverage:

    - **VADER** sentiment scoring  
    - **LDA** topic modelling (7 topics)  
    - **5 War Angles** breakdown  
    - **Articles vs Comments** comparison  
    - **Entity Tone** tracking  
    - **Media Framing** analysis  
    """)
    st.divider()
    st.caption(" Media Sentiment Analysis")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero-banner">
  <div class="hero-tag">📡 NLP Sentiment Intelligence Platform</div>
  <h1 class="hero-title">Media Framing &amp; Public Sentiment<br>US/Israel–Iran War Coverage</h1>
  <p class="hero-subtitle">VADER Sentiment · LDA Topic Modelling · Cross-Outlet Framing · Articles vs. Public Comments</p>
</div>
""", unsafe_allow_html=True)


if not articles_file or not comments_file:
    st.info("👈  Upload both CSV files in the sidebar to begin the analysis.")
    st.markdown("""
    **Expected files:**
    - `master_updated1.csv` → News articles (columns: `SOURCE`, `TEXT`)
    - `Master comments data1.csv` → Social media comments (columns: `source`, `comment_text`)
    """)
    st.stop()


# ─── Load & Process ───────────────────────────────────────────────────────────
with st.spinner("🔄 Loading and preprocessing data (this takes ~1-2 minutes the first time)…"):
    articles_raw, comments_raw = load_data(articles_file, comments_file)
    articles_df, comments_df   = preprocess_data(articles_raw, comments_raw)
    articles_df = score_articles(articles_df)
    comments_df = score_comments(comments_df)
    sentence_df = build_sentence_df(articles_df)
    art_df, com_df = build_combined_df(articles_df, comments_df)
    combined_df = pd.concat([art_df, com_df], ignore_index=True)
    topic_df, topic_assignments, topic_labels, lda_model, vectorizer, feature_names = run_lda(articles_df)
    articles_df['dominant_topic'] = [topic_labels[t] if t < len(topic_labels) else f'Topic {t+1}'
                                      for t in topic_assignments[:len(articles_df)]]


# ─── KPI Row ──────────────────────────────────────────────────────────────────
source_avg = sentence_df.groupby('source')['score'].mean().sort_values()
art_avg = art_df['score'].mean()
com_avg = com_df['score'].mean()
overall_neg_pct = round((sentence_df['label']=='Negative').mean()*100,1)

k1,k2,k3,k4,k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card"><p class="metric-number">{len(articles_df):,}</p><p class="metric-label">Articles Analysed</p></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><p class="metric-number">{len(comments_df):,}</p><p class="metric-label">Comments Scored</p></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card"><p class="metric-number">{len(sentence_df):,}</p><p class="metric-label">Sentences Tokenised</p></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><p class="metric-number">{overall_neg_pct}%</p><p class="metric-label">Negative Sentences</p></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="metric-card"><p class="metric-number">{len(articles_df['source'].unique())}</p><p class="metric-label">News Outlets</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Sentiment Overview",
    "🎯 5 War Angles",
    "🏛️ Topic Modelling",
    "👥 Articles vs Comments",
    "🔬 Entity Tone",
    "🗞️ Media Framing",
    "📋 Raw Data"
])


# ══════════════════════════════
# TAB 1 — SENTIMENT OVERVIEW
# ══════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-header">Sentiment Distribution Across Outlets</p>', unsafe_allow_html=True)

    # Key insights
    most_neg = source_avg.idxmin()
    most_pos = source_avg.idxmax()
    st.markdown(f"""
    <div class="insight-box warn">🔴 <b>Most Negative Outlet:</b> {most_neg} (avg score: {source_avg.min():.4f})</div>
    <div class="insight-box">🟢 <b>Most Positive Outlet:</b> {most_pos} (avg score: {source_avg.max():.4f})</div>
    """, unsafe_allow_html=True)

    # Chart 1 — Sentiment dist by outlet (articles)
    label_counts = pd.crosstab(sentence_df['source'], sentence_df['label'])
    cols_order = [c for c in ['Negative','Neutral','Positive'] if c in label_counts.columns]
    chart_data = label_counts[cols_order].reset_index()
    outlet_col = chart_data.columns[0]
    chart_data.rename(columns={outlet_col:'News Outlet'}, inplace=True)
    df_melted = chart_data.melt(id_vars='News Outlet', value_vars=cols_order,
                                 var_name='Sentiment', value_name='Number of Sentences')

    fig1 = px.bar(df_melted, x='News Outlet', y='Number of Sentences', color='Sentiment',
                   barmode='group', color_discrete_map=COLOR_SENT,
                   category_orders={'Sentiment':['Negative','Neutral','Positive']},
                   title='Sentiment Distribution Across News Outlets — Article Sentences',
                   text_auto=True)
    fig1.update_traces(textposition='outside', textfont_size=9)
    fig1.update_layout(xaxis_tickangle=-30, bargap=0.2, bargroupgap=0.05, height=450)
    plotly_defaults(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 — Avg sentiment per outlet
    st.markdown('<p class="section-header">Average VADER Score by Outlet</p>', unsafe_allow_html=True)
    df_avg = source_avg.reset_index()
    df_avg.columns = ['News Outlet','Avg Sentiment Score']
    colors = ['#2ECC71' if v>=0 else '#E74C3C' for v in df_avg['Avg Sentiment Score']]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_avg['News Outlet'], y=df_avg['Avg Sentiment Score'].abs(),
                           text=df_avg['Avg Sentiment Score'].round(4), textposition='outside',
                           marker_color=colors, marker_line_color='white', marker_line_width=1,
                           hovertemplate='<b>%{x}</b><br>Avg Score: %{text}<extra></extra>', width=0.6))
    fig2.add_trace(go.Bar(x=[None],y=[None],marker_color='#2ECC71',name='Positive',showlegend=True))
    fig2.add_trace(go.Bar(x=[None],y=[None],marker_color='#E74C3C',name='Negative',showlegend=True))
    fig2.update_layout(title='Average Sentiment Score by News Outlet',
                        xaxis=dict(title='News Outlet',tickangle=-30),
                        yaxis=dict(title='Avg VADER Score (Absolute)'), height=420, showlegend=True)
    plotly_defaults(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # Comments sentiment
    st.markdown('<p class="section-header">Comment Sentiment by Platform</p>', unsafe_allow_html=True)
    com_label_counts = pd.crosstab(comments_df['source'], comments_df['sentiment_label'])
    com_cols = [c for c in ['Negative','Neutral','Positive'] if c in com_label_counts.columns]
    com_melted = com_label_counts[com_cols].reset_index().melt(id_vars='source', value_vars=com_cols,
                                                                var_name='Sentiment', value_name='Count')
    fig3 = px.bar(com_melted, x='source', y='Count', color='Sentiment', barmode='group',
                   color_discrete_map=COLOR_SENT, text_auto=True,
                   title='Sentiment Distribution Across Social Media Platforms')
    fig3.update_traces(textposition='outside', textfont_size=9)
    fig3.update_layout(xaxis_title='Platform', yaxis_title='Number of Comments',
                        xaxis_tickangle=-20, height=420)
    plotly_defaults(fig3)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════
# TAB 2 — 5 WAR ANGLES
# ══════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-header">Sentiment Across the 5 War Coverage Angles</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box info">
    Each war angle is identified by matching sentences against a curated keyword set.
    Scores reflect the average VADER compound score for matching sentences.
    </div>
    """, unsafe_allow_html=True)

    angle_results = []
    for angle_name, keywords in ANGLES.items():
        pattern = '|'.join([rf'\b{re.escape(k)}\b' for k in keywords])
        subset = sentence_df[sentence_df['sentence'].str.contains(pattern, case=False, na=False)]
        if len(subset) > 0:
            angle_results.append({
                'Angle': angle_name, 'Sentences Found': len(subset),
                'Avg Score': round(subset['score'].mean(), 4),
                '% Positive': round((subset['label']=='Positive').mean()*100,1),
                '% Neutral':  round((subset['label']=='Neutral').mean()*100,1),
                '% Negative': round((subset['label']=='Negative').mean()*100,1)
            })
    angle_df = pd.DataFrame(angle_results)

    # Display table
    st.dataframe(angle_df.style.background_gradient(subset=['Avg Score'], cmap='RdYlGn', vmin=-0.3, vmax=0.3),
                 use_container_width=True, hide_index=True)

    # Dual chart
    fig_ang = make_subplots(rows=1, cols=2,
                             subplot_titles=('Average Sentiment Score by Angle',
                                             'Sentiment Distribution (%) by Angle'))
    colors_a = ['#2ECC71' if v>=0 else '#E74C3C' for v in angle_df['Avg Score']]
    fig_ang.add_trace(go.Bar(x=angle_df['Angle'], y=angle_df['Avg Score'].abs(),
                              text=angle_df['Avg Score'].round(3), textposition='outside',
                              marker_color=colors_a, cliponaxis=False, showlegend=False, name='Avg Score'),
                       row=1, col=1)
    for sent, color in [('% Negative','#E74C3C'),('% Neutral','#95A5A6'),('% Positive','#2ECC71')]:
        lbl = sent.replace('% ','')
        fig_ang.add_trace(go.Bar(x=angle_df['Angle'], y=angle_df[sent], name=lbl,
                                  marker_color=color,
                                  text=angle_df[sent].round(1).astype(str)+'%', textposition='inside',
                                  textfont=dict(size=9)), row=1, col=2)
    fig_ang.update_layout(barmode='stack', height=480, legend=dict(title='Sentiment'),
                           title='Sentiment Analysis Across 5 War Coverage Angles')
    fig_ang.update_xaxes(tickangle=-25)
    fig_ang.update_yaxes(title_text='VADER Score (Absolute)', row=1, col=1)
    fig_ang.update_yaxes(title_text='Percentage (%)', range=[0,105], row=1, col=2)
    plotly_defaults(fig_ang)
    st.plotly_chart(fig_ang, use_container_width=True)

    # Cross-outlet × angle heatmap
    st.markdown('<p class="section-header">Angle × Outlet Heatmap</p>', unsafe_allow_html=True)
    cross_results = []
    for angle_name, keywords in ANGLES.items():
        pattern = '|'.join([rf'\b{re.escape(k)}\b' for k in keywords])
        subset = sentence_df[sentence_df['sentence'].str.contains(pattern, case=False, na=False)]
        for src, grp in subset.groupby('source'):
            cross_results.append({'Angle':angle_name,'Outlet':src,
                                   'Avg Score':round(grp['score'].mean(),4)})
    cross_df = pd.DataFrame(cross_results)
    pivot = cross_df.pivot_table(index='Outlet', columns='Angle', values='Avg Score').round(3)

    fig_heat = px.imshow(pivot, color_continuous_scale='RdYlGn', zmin=-0.3, zmax=0.3,
                          text_auto=True, title='Average Sentiment Score per Outlet per Angle',
                          aspect='auto')
    fig_heat.update_layout(height=420)
    plotly_defaults(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════
# TAB 3 — TOPIC MODELLING
# ══════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-header">LDA Topic Modelling (7 Topics)</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box info">
    Latent Dirichlet Allocation (LDA) was run on preprocessed article texts to uncover
    7 latent themes. Each article is assigned to its dominant topic.
    </div>
    """, unsafe_allow_html=True)

    # Topic table
    st.dataframe(topic_df, use_container_width=True, hide_index=True)

    # Topic distribution chart
    topic_counts = articles_df['dominant_topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic','Number of Articles']
    fig_topic = px.bar(topic_counts, x='Topic', y='Number of Articles', text='Number of Articles',
                        color='Topic', color_discrete_sequence=px.colors.qualitative.Bold,
                        title='Distribution of Topics Across Articles (LDA — 7 Topics)')
    fig_topic.update_traces(textposition='outside', cliponaxis=False)
    fig_topic.update_layout(showlegend=False, xaxis_tickangle=-30, height=480)
    plotly_defaults(fig_topic)
    st.plotly_chart(fig_topic, use_container_width=True)

    # Word clouds
    st.markdown('<p class="section-header">Topic Word Clouds</p>', unsafe_allow_html=True)
    try:
        from wordcloud import WordCloud
        fig_wc, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor='white')
        axes = axes.flatten()
        for idx, topic in enumerate(lda_model.components_):
            word_freq = {feature_names[i]: topic[i] for i in range(len(feature_names))}
            wc = WordCloud(width=600, height=300, background_color='white',
                            colormap='plasma').generate_from_frequencies(word_freq)
            axes[idx].imshow(wc)
            axes[idx].axis('off')
            lbl = topic_labels[idx] if idx < len(topic_labels) else f'Topic {idx+1}'
            axes[idx].set_title(lbl, fontsize=10, fontweight='bold', color='#2d3748')
        axes[-1].axis('off')
        plt.tight_layout()
        st.pyplot(fig_wc)
        plt.close()
    except ImportError:
        st.warning("WordCloud library not installed. Run `pip install wordcloud` to enable word clouds.")


# ══════════════════════════════
# TAB 4 — ARTICLES VS COMMENTS
# ══════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-header">Overall: News Articles vs Public Comments</p>', unsafe_allow_html=True)

    overall = combined_df.groupby('data_type').agg(
        Total_Texts=('score','count'), Avg_Score=('score','mean'), Std_Dev=('score','std')
    ).round(4)
    label_dist = combined_df.groupby(['data_type','label']).size().unstack(fill_value=0)
    label_pct  = label_dist.div(label_dist.sum(axis=1), axis=0).mul(100).round(1)
    label_pct.columns = [f'% {c}' for c in label_pct.columns]
    comparison_table = pd.concat([overall, label_pct], axis=1)

    # Insight
    gap = abs(art_avg - com_avg)
    more_neg = "Comments are MORE NEGATIVE" if com_avg < art_avg else "Articles are MORE NEGATIVE"
    st.markdown(f"""
    <div class="insight-box warn">
    📊 <b>{more_neg}</b> — gap of {gap:.4f} VADER points between professional journalism and public commentary.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(comparison_table.style.background_gradient(subset=['Avg_Score'], cmap='RdYlGn', vmin=-0.2, vmax=0.2),
                 use_container_width=True)

    # Dual chart
    types  = comparison_table.index.tolist()
    scores = comparison_table['Avg_Score'].tolist()
    c_bar  = ['#2ECC71' if s>=0 else '#E74C3C' for s in scores]
    score_df2 = pd.DataFrame({'Type':types,'Avg Score':scores})
    cols_pct = [c for c in ['% Negative','% Neutral','% Positive'] if c in comparison_table.columns]
    label_df2 = pd.DataFrame({'Type':types,
                               '% Negative': comparison_table['% Negative'].values if '% Negative' in comparison_table else [0]*len(types),
                               '% Neutral':  comparison_table['% Neutral'].values  if '% Neutral'  in comparison_table else [0]*len(types),
                               '% Positive': comparison_table['% Positive'].values if '% Positive' in comparison_table else [0]*len(types)
                               }).melt(id_vars='Type',var_name='Sentiment',value_name='Percentage')

    fig_comp = make_subplots(rows=1, cols=2,
                              subplot_titles=('Average Sentiment Score','Sentiment Distribution (%)'))
    fig_comp.add_trace(go.Bar(x=score_df2['Type'], y=score_df2['Avg Score'].abs(),
                               text=score_df2['Avg Score'].round(4), textposition='outside',
                               marker_color=c_bar, cliponaxis=False, showlegend=False, width=0.5), row=1, col=1)
    for sent, color in [('% Negative','#E74C3C'),('% Neutral','#95A5A6'),('% Positive','#2ECC71')]:
        lbl = sent.replace('% ','')
        sub = label_df2[label_df2['Sentiment']==sent]
        fig_comp.add_trace(go.Bar(x=sub['Type'], y=sub['Percentage'], name=lbl, marker_color=color,
                                   text=sub['Percentage'].round(1).astype(str)+'%',
                                   textposition='outside', cliponaxis=False), row=1, col=2)
    fig_comp.update_layout(barmode='group', height=450, legend=dict(title='Sentiment'),
                            title='News Articles vs Comments — Sentiment Comparison')
    fig_comp.update_yaxes(title_text='VADER Score (Absolute)', row=1, col=1)
    fig_comp.update_yaxes(title_text='Percentage (%)', row=1, col=2)
    plotly_defaults(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

    # By source
    st.markdown('<p class="section-header">Sentiment by Source: Articles vs Comments</p>', unsafe_allow_html=True)
    src_comp = combined_df.groupby(['source','data_type'])['score'].mean().round(4).reset_index()
    src_comp.columns = ['Source','Data Type','Avg Sentiment Score']
    src_comp['Abs Score'] = src_comp['Avg Sentiment Score'].abs()
    fig_src = px.bar(src_comp, x='Source', y='Abs Score', color='Data Type', barmode='group',
                      text=src_comp['Avg Sentiment Score'].round(4),
                      color_discrete_map=COLOR_TYPE,
                      title='Sentiment Score by Source: News Articles vs Comments')
    fig_src.update_traces(textposition='outside', cliponaxis=False)
    fig_src.update_layout(xaxis_tickangle=-30, height=450)
    plotly_defaults(fig_src)
    st.plotly_chart(fig_src, use_container_width=True)

    # 5 Angles comparison
    st.markdown('<p class="section-header">5 War Angles: Articles vs Comments</p>', unsafe_allow_html=True)
    angles_v2 = {
        'i. Military Operations': ['missile','drone','airstrike','strike','bomb','military','attack','weapon','navy','fighter','irgc','centcom','operation','warship','carrier','troops','ballistic'],
        'ii. Geopolitical Tensions': ['sanction','ally','alliance','tension','gulf','ceasefire','diplomacy','china','russia','europe','nato','un','sovereignty','escalat','neutral','mediat'],
        'iii. Economic Impact': ['oil','energy','price','market','barrel','gas','hormuz','inflation','economy','supply','trade','recession','lng','fuel','cost','crude','shipping','tanker'],
        'iv. Media Narratives': ['propaganda','misinformation','bias','narrative','fake','disinformation','media','distrust','mislead','manipulate','youtube','facebook','report','coverage'],
        'v. Support for War': ['support','oppose','protest','condemn','approve','justify','coalition','against','favor','opposition','opinion','public','poll','stance','position']
    }
    angle_comp = []
    for ang, kws in angles_v2.items():
        patt = '|'.join([rf'\b{re.escape(k)}\b' for k in kws])
        for dtype in ['News Article','Comment']:
            sub = combined_df[(combined_df['data_type']==dtype) &
                               (combined_df['text'].str.contains(patt, case=False, na=False))]
            if len(sub)>0:
                angle_comp.append({'Angle':ang,'Data Type':dtype,'Count':len(sub),
                                    'Avg Score':round(sub['score'].mean(),4)})
    acd = pd.DataFrame(angle_comp)
    acd['Abs Score'] = acd['Avg Score'].abs()
    fig_acomp = px.bar(acd, x='Angle', y='Abs Score', color='Data Type', barmode='group',
                        text=acd['Avg Score'].round(3), color_discrete_map=COLOR_TYPE,
                        title='Sentiment Across 5 War Angles: Articles vs Comments',
                        category_orders={'Angle':list(angles_v2.keys())})
    fig_acomp.update_traces(textposition='outside', cliponaxis=False)
    fig_acomp.update_layout(xaxis_tickangle=-25, height=450)
    plotly_defaults(fig_acomp)
    st.plotly_chart(fig_acomp, use_container_width=True)

    # Top words
    st.markdown('<p class="section-header">Most Frequent Words</p>', unsafe_allow_html=True)
    art_words = ' '.join(articles_df['clean_text'].dropna()).split()
    com_words = ' '.join(comments_df['clean_text'].dropna()).split()
    art_top = pd.DataFrame(Counter(art_words).most_common(15), columns=['Word','Count'])
    com_top = pd.DataFrame(Counter(com_words).most_common(15), columns=['Word','Count'])

    col_a, col_b = st.columns(2)
    with col_a:
        fig_aw = px.bar(art_top[::-1], x='Count', y='Word', orientation='h',
                         color_discrete_sequence=['#3498DB'],
                         title='Top 15 Words — News Articles')
        fig_aw.update_layout(height=430, yaxis=dict(autorange=True))
        plotly_defaults(fig_aw)
        st.plotly_chart(fig_aw, use_container_width=True)
    with col_b:
        fig_cw = px.bar(com_top[::-1], x='Count', y='Word', orientation='h',
                         color_discrete_sequence=['#E67E22'],
                         title='Top 15 Words — Comments')
        fig_cw.update_layout(height=430, yaxis=dict(autorange=True))
        plotly_defaults(fig_cw)
        st.plotly_chart(fig_cw, use_container_width=True)

    # Box plot
    st.markdown('<p class="section-header">Score Distribution (Box Plot)</p>', unsafe_allow_html=True)
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=art_df['score'].tolist(), name='News Articles',
                              marker_color='#3498DB', boxmean=True))
    fig_box.add_trace(go.Box(y=com_df['score'].tolist(), name='Comments',
                              marker_color='#E67E22', boxmean=True))
    fig_box.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig_box.update_layout(title='Sentiment Score Distribution: Articles vs Comments',
                           yaxis_title='VADER Compound Score', height=430)
    plotly_defaults(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════
# TAB 5 — ENTITY TONE
# ══════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-header">Tone Toward Key Entities (Iran, Israel, US, Trump)</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box info">
    Entity tone is measured by extracting all sentences mentioning each entity
    and computing the average VADER compound score across those sentences.
    </div>
    """, unsafe_allow_html=True)

    # Article entity analysis
    entity_results = []
    for ent in ENTITIES:
        sub = sentence_df[sentence_df['sentence'].str.contains(rf'\b{ent}\b', case=False, na=False)]
        if len(sub)>0:
            entity_results.append({'Entity':ent.upper(),'Sentences':len(sub),
                                    'Avg Score':round(sub['score'].mean(),4),
                                    '% Positive':round((sub['label']=='Positive').mean()*100,1),
                                    '% Neutral': round((sub['label']=='Neutral').mean()*100,1),
                                    '% Negative':round((sub['label']=='Negative').mean()*100,1)})
    entity_df = pd.DataFrame(entity_results)

    st.dataframe(entity_df.style.background_gradient(subset=['Avg Score'], cmap='RdYlGn', vmin=-0.3, vmax=0.3),
                 use_container_width=True, hide_index=True)

    # Entity dual chart
    fig_ent = make_subplots(rows=1, cols=2,
                             subplot_titles=('Average Media Tone Toward Key Entities',
                                             'Sentiment Breakdown (%) per Entity'))
    col_e = ['#2ECC71' if v>=0 else '#E74C3C' for v in entity_df['Avg Score']]
    hover_e = [f"<b>{r['Entity']}</b><br>Avg: {r['Avg Score']}<br>Sentences: {r['Sentences']}<br>+{r['% Positive']}% | ={r['% Neutral']}% | -{r['% Negative']}%"
               for _,r in entity_df.iterrows()]
    fig_ent.add_trace(go.Bar(x=entity_df['Entity'], y=entity_df['Avg Score'].abs(),
                              text=entity_df['Avg Score'].round(3), textposition='outside',
                              marker_color=col_e, hovertext=hover_e, hoverinfo='text',
                              cliponaxis=False, width=0.6, showlegend=False), row=1, col=1)
    for sent, color in [('% Negative','#E74C3C'),('% Neutral','#95A5A6'),('% Positive','#2ECC71')]:
        lbl = sent.replace('% ','')
        fig_ent.add_trace(go.Bar(x=entity_df['Entity'], y=entity_df[sent], name=lbl,
                                  marker_color=color,
                                  text=entity_df[sent].round(1).astype(str)+'%', textposition='inside',
                                  textfont=dict(size=9)), row=1, col=2)
    fig_ent.add_trace(go.Bar(x=[None],y=[None],marker_color='#2ECC71',name='Positive',showlegend=True))
    fig_ent.add_trace(go.Bar(x=[None],y=[None],marker_color='#E74C3C',name='Negative',showlegend=True))
    fig_ent.update_layout(barmode='stack', height=480, legend=dict(title='Sentiment'),
                           title='Media Tone Toward Key Entities — US/Israel-Iran War')
    fig_ent.update_xaxes(tickangle=-15)
    fig_ent.update_yaxes(title_text='VADER Score (Absolute)', row=1, col=1)
    fig_ent.update_yaxes(title_text='Percentage (%)', range=[0,105], row=1, col=2)
    plotly_defaults(fig_ent)
    st.plotly_chart(fig_ent, use_container_width=True)

    # Entity: Articles vs Comments
    st.markdown('<p class="section-header">Entity Tone: Articles vs Comments</p>', unsafe_allow_html=True)
    ent_comp = []
    for ent in ENTITIES:
        for dtype in ['News Article','Comment']:
            sub = combined_df[(combined_df['data_type']==dtype) &
                               (combined_df['text'].str.contains(rf'\b{ent}\b', case=False, na=False))]
            if len(sub)>0:
                ent_comp.append({'Entity':ent.upper(),'Data Type':dtype,
                                  'Count':len(sub),'Avg Score':round(sub['score'].mean(),4)})
    ecd = pd.DataFrame(ent_comp)
    ecd['Abs Score'] = ecd['Avg Score'].abs()
    fig_ec = px.bar(ecd, x='Entity', y='Abs Score', color='Data Type', barmode='group',
                     text=ecd['Avg Score'].round(4), color_discrete_map=COLOR_TYPE,
                     title='Tone Toward Key Entities: News Articles vs Comments',
                     category_orders={'Entity':[e.upper() for e in ENTITIES]})
    fig_ec.update_traces(textposition='outside', cliponaxis=False)
    fig_ec.update_layout(height=450)
    plotly_defaults(fig_ec)
    st.plotly_chart(fig_ec, use_container_width=True)


# ══════════════════════════════
# TAB 6 — MEDIA FRAMING
# ══════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-header">Media Framing Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box info">
    News framing theory suggests that how events are described shapes public perception.
    Four key frames are detected by counting domain-specific keyword frequencies in article text.
    </div>
    """, unsafe_allow_html=True)

    all_words = ' '.join(articles_df['clean_text'].dropna()).split()
    word_counts = Counter(all_words)
    frame_df = pd.DataFrame([
        {'Frame':frame,'Keyword Mentions':sum(word_counts.get(w,0) for w in kws)}
        for frame, kws in FRAMES.items()
    ]).sort_values('Keyword Mentions', ascending=False)

    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.dataframe(frame_df, use_container_width=True, hide_index=True)
        dominant = frame_df.iloc[0]
        st.markdown(f"""
        <div class="insight-box">
        🏆 <b>Dominant Frame:</b> {dominant['Frame']} with {dominant['Keyword Mentions']:,} keyword mentions.
        This suggests coverage is primarily framed around <b>{dominant['Frame'].lower().replace(' frame','')}</b> language.
        </div>
        """, unsafe_allow_html=True)
    with c_right:
        fig_frame = px.pie(frame_df, names='Frame', values='Keyword Mentions',
                            color_discrete_sequence=['#E74C3C','#E67E22','#3498DB','#9B59B6'],
                            title='Media Frame Share')
        fig_frame.update_traces(textposition='inside', textinfo='percent+label')
        fig_frame.update_layout(height=350, showlegend=False)
        plotly_defaults(fig_frame)
        st.plotly_chart(fig_frame, use_container_width=True)

    fig_frame_bar = px.bar(frame_df, x='Frame', y='Keyword Mentions', text='Keyword Mentions',
                            color='Frame',
                            color_discrete_sequence=['#E74C3C','#E67E22','#3498DB','#9B59B6'],
                            title='Media Framing of the Iran War — Keyword Frequency by Frame')
    fig_frame_bar.update_traces(textposition='outside', cliponaxis=False)
    fig_frame_bar.update_layout(showlegend=False, xaxis_tickangle=-15, height=420)
    plotly_defaults(fig_frame_bar)
    st.plotly_chart(fig_frame_bar, use_container_width=True)


# ══════════════════════════════
# TAB 7 — RAW DATA
# ══════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-header">Raw & Processed Data Explorer</p>', unsafe_allow_html=True)

    data_choice = st.selectbox("Select dataset to explore:",
                                ["Articles (with sentiment)", "Comments (with sentiment)",
                                 "Sentence-level Scores", "Topic Model Results"])

    if data_choice == "Articles (with sentiment)":
        st.dataframe(articles_df[['source','text','clean_text','sentiment_score','sentiment_label','dominant_topic']],
                     use_container_width=True)
        csv = articles_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Articles CSV", csv, "articles_sentiment.csv", "text/csv")

    elif data_choice == "Comments (with sentiment)":
        st.dataframe(comments_df[['source','text','clean_text','sentiment_score','sentiment_label']],
                     use_container_width=True)
        csv = comments_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Comments CSV", csv, "comments_sentiment.csv", "text/csv")

    elif data_choice == "Sentence-level Scores":
        st.dataframe(sentence_df, use_container_width=True)
        csv = sentence_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sentences CSV", csv, "sentence_scores.csv", "text/csv")

    elif data_choice == "Topic Model Results":
        st.dataframe(topic_df, use_container_width=True, hide_index=True)
        csv = topic_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Topics CSV", csv, "lda_topics.csv", "text/csv")

    # Final summary
    st.divider()
    st.markdown('<p class="section-header">📋 Final Analysis Summary</p>', unsafe_allow_html=True)

    overall_dist = sentence_df['label'].value_counts(normalize=True).mul(100).round(1)
    s1,s2,s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        **Overall Sentence Sentiment**
        - 🔴 Negative: {overall_dist.get('Negative',0)}%
        - ⚪ Neutral: {overall_dist.get('Neutral',0)}%
        - 🟢 Positive: {overall_dist.get('Positive',0)}%
        """)
    with s2:
        st.markdown(f"""
        **Outlet Rankings**
        - 🔴 Most Negative: **{source_avg.idxmin()}** ({source_avg.min():.4f})
        - 🟢 Most Positive: **{source_avg.idxmax()}** ({source_avg.max():.4f})
        """)
    with s3:
        entity_results_s = []
        for ent in ENTITIES:
            sub = sentence_df[sentence_df['sentence'].str.contains(rf'\b{ent}\b', case=False, na=False)]
            if len(sub)>0:
                entity_results_s.append({'Entity':ent.upper(),'Sentences':len(sub)})
        edf_s = pd.DataFrame(entity_results_s)
        top_ent = edf_s.loc[edf_s['Sentences'].idxmax(),'Entity'] if len(edf_s)>0 else 'N/A'
        dominant_frame = frame_df.iloc[0]['Frame']
        st.markdown(f"""
        **Key Findings**
        - 📣 Most Mentioned Entity: **{top_ent}**
        - 🗞️ Dominant Media Frame: **{dominant_frame}**
        - 📊 Articles avg score: **{art_avg:.4f}**
        - 💬 Comments avg score: **{com_avg:.4f}**
        """)
