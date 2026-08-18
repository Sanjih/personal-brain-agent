import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Motion AI Coach", page_icon="🥊", layout="wide")

# Style CSS personnalisé pour l'UX
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; }
    .advice-card-urgent { background-color: #3b1111; border-left: 5px solid #ff4b4b; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    .advice-card-warning { background-color: #3b2b11; border-left: 5px solid #ffa100; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    .advice-card-good { background-color: #113b1b; border-left: 5px solid #00c853; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Motion AI Studio — Votre Coach Personnel")

# Barre latérale : Clé API & Solde
st.sidebar.header("👤 Mon Compte")
api_key = st.sidebar.text_input("Clé API", value="secret123", type="password")

headers = {"x-api-key": api_key}
res = requests.get(f"{API_URL}/users/balance", headers=headers)

if res.status_code == 200:
    balance = res.json()["balance_minutes"]
    st.sidebar.metric("Solde Crédits", f"{balance} min")
else:
    st.sidebar.error("Clé API non reconnue")

# Choix de la discipline
st.subheader("1. Choisissez votre discipline")
col_mode1, col_mode2 = st.columns(2)
with col_mode1:
    mode_combat = st.checkbox("🥊 Sports de Combat (Boxe/MMA)", value=True)
with col_mode2:
    mode_dance = st.checkbox("💃 Danse & Chorégraphie", value=not mode_combat)

selected_mode = "combat" if mode_combat else "dance"

# Drag and drop vidéo
st.subheader("2. Importer votre vidéo")
uploaded_file = st.file_uploader("Glissez-déposez votre vidéo ici (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("🚀 Analyser ma prestation", type="primary", use_container_width=True):
        endpoint = f"/analyze/{selected_mode}"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
        
        with st.spinner("L'IA étudie votre posture et vos mouvements..."):
            response = requests.post(f"{API_URL}{endpoint}", headers=headers, files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.balloons()
                
                st.markdown("---")
                st.subheader("📊 Rapport du Coach")
                
                # Haut de page : Score et Métriques
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Score Technique", f"{data['performance_score']} / 100")
                col2.metric("Durée Déduite", f"{data['duration_minutes']} min")
                col3.metric("Solde Restant", f"{data['remaining_balance_minutes']} min")
                col4.metric("Frames Analysées", data['frames_analyzed'])
                
                st.markdown("---")
                
                # Section 2 colonnes : Conseils à gauche, Vidéo à droite
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("💡 Conseils d'Amélioration")
                    for advice in data["coach_advices"]:
                        adv_type = advice["type"]
                        text = advice["text"]
                        
                        if adv_type == "urgent":
                            st.markdown(f'<div class="advice-card-urgent">🔴 <b>Axe prioritaire :</b> {text}</div>', unsafe_allow_html=True)
                        elif adv_type == "warning":
                            st.markdown(f'<div class="advice-card-warning">🟡 <b>À corriger :</b> {text}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="advice-card-good">🟢 <b>Point fort :</b> {text}</div>', unsafe_allow_html=True)
                
                with col_right:
                    st.subheader("📹 Analyse Visuelle Squelettique")
                    video_url = f"{API_URL}{data['annotated_video_url']}"
                    st.video(video_url)
                    
            elif response.status_code == 402:
                st.error("Solde insuffisant ! Rechargez vos minutes pour continuer.")
            else:
                st.error(f"Erreur : {response.json().get('detail')}")