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
res = requests.get(f"{API_URL}/users/balance", headers=headers)

if res.status_code == 200:
    balance = res.json()["balance_minutes"]
    st.sidebar.metric("Solde Crédits", f"{balance} min")
else:
    st.sidebar.error("Clé API invalide")

st.subheader("1. Discipline")
col_mode1, col_mode2 = st.columns(2)
with col_mode1:
    mode_combat = st.checkbox("🥊 Sports de Combat", value=True)
with col_mode2:
    mode_dance = st.checkbox("💃 Danse & Chorégraphie", value=not mode_combat)

selected_mode = "combat" if mode_combat else "dance"

st.subheader("2. Importer la vidéo")
uploaded_file = st.file_uploader("Choisissez un fichier vidéo", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("🚀 Générer la Timeline du Coach", type="primary", use_container_width=True):
        endpoint = f"/analyze/{selected_mode}"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
        
        with st.spinner("Analyse image par image en cours..."):
            response = requests.post(f"{API_URL}{endpoint}", headers=headers, files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.balloons()
                
                st.markdown("---")
                
                # Métriques haut de page
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Score Technique", f"{data['performance_score']} / 100")
                col2.metric("Durée Analyse", f"{data['duration_minutes']} min")
                col3.metric("Solde Restant", f"{data['remaining_balance_minutes']} min")
                col4.metric("Frames Traitées", data['frames_analyzed'])
                
                st.markdown("---")
                
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("⏱️ Chronologie des Remarques (Timeline)")
                    
                    for event in data["timeline_events"]:
                        t_str = event["timestamp"]
                        msg = event["message"]
                        e_type = event["type"]
                        
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
                    video_url = f"{API_URL}{data['annotated_video_url']}"
                    st.video(video_url)
                    
            else:
                st.error(f"Erreur : {response.json().get('detail')}")