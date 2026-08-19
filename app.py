import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Motion AI Coach", page_icon="⏱️", layout="wide")

# CSS pour le style de la Timeline
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; }
    
    .timeline-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding: 10px 15px;
        border-radius: 8px;
        font-family: monospace;
    }
    .time-badge {
        background-color: #262730;
        color: #00d4ff;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        margin-right: 15px;
        border: 1px solid #00d4ff;
    }
    .bg-urgent { background-color: #3b1111; border-left: 4px solid #ff4b4b; }
    .bg-warning { background-color: #3b2b11; border-left: 4px solid #ffa100; }
    .bg-good { background-color: #113b1b; border-left: 4px solid #00c853; }
    </style>
""", unsafe_allow_html=True)

st.title("⏱️ Motion AI Studio — Analyse Temporelle")

st.sidebar.header("👤 Mon Compte")
api_key = st.sidebar.text_input("Clé API", value="secret123", type="password")
headers = {"x-api-key": api_key}

# Vérification du solde avec gestion des erreurs de connexion
try:
    res = requests.get(f"{API_URL}/users/balance", headers=headers, timeout=5)
    if res.status_code == 200:
        balance = res.json().get("balance_minutes", 0)
        st.sidebar.metric("Solde Crédits", f"{balance} min")
    else:
        st.sidebar.error("Clé API invalide")
except Exception as e:
    st.sidebar.error(f"Backend inaccessible : {e}")

st.subheader("1. Discipline")
discipline = st.radio(
    "Choisissez la discipline :",
    ["🥊 Sports de Combat", "💃 Danse & Chorégraphie"],
    horizontal=True
)
selected_mode = "combat" if "Combat" in discipline else "dance"

st.subheader("2. Importer la vidéo")
uploaded_file = st.file_uploader("Choisissez un fichier vidéo", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("🚀 Générer la Timeline du Coach", type="primary", use_container_width=True):
        endpoint = f"/analyze/{selected_mode}"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
        
        with st.spinner("Analyse image par image en cours (YOLO + Nebius VLM)..."):
            try:
                # Requête vers FastAPI avec un timeout long pour le traitement vidéo
                response = requests.post(
                    f"{API_URL}{endpoint}", 
                    headers=headers, 
                    files=files,
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.balloons()
                    st.markdown("---")
                    
                    # Métriques haut de page
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Score Technique", f"{data.get('performance_score', 0)} / 100")
                    col2.metric("Durée Analyse", f"{data.get('duration_minutes', 0)} min")
                    col3.metric("Solde Restant", f"{data.get('remaining_balance_minutes', 0)} min")
                    col4.metric("Frames Traitées", data.get('frames_analyzed', 0))
                    
                    st.markdown("---")
                    
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.subheader("⏱️ Chronologie des Remarques (Timeline)")
                        for event in data.get("timeline_events", []):
                            t_str = event.get("timestamp", "00:00")
                            msg = event.get("message", "")
                            e_type = event.get("type", "good")
                            
                            css_class = "bg-urgent" if e_type == "urgent" else ("bg-warning" if e_type == "warning" else "bg-good")
                            icon = "🔴" if e_type == "urgent" else ("🟡" if e_type == "warning" else "🟢")
                            
                            st.markdown(f'''
                                <div class="timeline-item {css_class}">
                                    <span class="time-badge">⏱️ {t_str}</span>
                                    <span>{icon} {msg}</span>
                                </div>
                            ''', unsafe_allow_html=True)
                    
                    with col_right:
                        st.subheader("📹 Rendu Vidéo Annoté")
                        video_path = data.get('annotated_video_url', '')
                        if video_path:
                            st.video(f"{API_URL}{video_path}")
                        else:
                            st.info("Aucune vidéo annotée renvoyée par le backend.")
                else:
                    # Affichage précis du statut d'erreur renvoyé par FastAPI
                    st.error(f"Erreur HTTP {response.status_code} : {response.text}")

            except requests.exceptions.Timeout:
                st.error("Le traitement a dépassé le temps limite (Timeout).")
            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter FastAPI. Vérifiez que 'uvicorn main:app' tourne bien.")
            except Exception as err:
                st.error(f"Une erreur inattendue est survenue : {err}")
