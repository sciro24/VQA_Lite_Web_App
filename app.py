import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ------------------------
# Config minimale (riuso path dal notebook)
# ------------------------
cfg = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': {
        'question_dim': 384,
        'image_feature_dim': 256,
        'attention_hidden_dim': 128,
        'dropout': 0.3,
    },
    'answers': ["aeroplano", "automobile", "uccello", "gatto", "cervo", "cane", "rana", "cavallo", "nave", "camion"],
    'paths': {
        'model_save_path': str(Path('models') / 'vqa_model_best.pth')
    },
    'embedding_model': 'all-MiniLM-L6-v2'
}

DEVICE = cfg['device']

# Ensure models directory exists
Path('models').mkdir(exist_ok=True)


def get_image_transform(is_training: bool = False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])


class VQANet(nn.Module):
    def __init__(self, num_answers, question_dim, image_feature_dim, attention_hidden_dim, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, image_feature_dim, kernel_size=1)
        self.attention_conv = nn.Conv2d(image_feature_dim + question_dim, attention_hidden_dim, 1)
        self.attention_fc = nn.Conv2d(attention_hidden_dim, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + question_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_hidden_dim, num_answers)
        )

    def forward(self, image, question_emb, temperature: float = 1.0):
        x = self.backbone(image)
        img_features = self.proj(x)
        B, C, H, W = img_features.shape
        question_emb_expanded = question_emb.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        combined_features = torch.cat([img_features, question_emb_expanded], dim=1)
        attn_hidden = torch.tanh(self.attention_conv(combined_features))
        logits = self.attention_fc(attn_hidden).view(B, -1)
        logits = logits / max(temperature, 1e-6)
        attn_weights = F.softmax(logits, dim=1).view(B, 1, H, W)
        attended_img_vector = (attn_weights * img_features).sum(dim=[2, 3])
        final_combined = torch.cat([attended_img_vector, question_emb], dim=1)
        return self.fc(final_combined)


@st.cache_resource
def load_embedding_model(name: str, device: str):
    return SentenceTransformer(name, device=device)


@st.cache_resource
def load_vqa_model(path: str, device: str):
    m_cfg = cfg['model']
    num_answers = len(cfg['answers'])
    model = VQANet(num_answers, m_cfg['question_dim'], m_cfg['image_feature_dim'], m_cfg['attention_hidden_dim'], dropout=m_cfg.get('dropout', 0.3))
    model.to(device)
    p = Path(path)
    if p.exists():
        try:
            model.load_state_dict(torch.load(p, map_location=device))
            # Note: we avoid using st.sidebar here because this function can be cached and
            # called outside of a Streamlit render context; UI notifications are handled in main.
        except Exception:
            pass
    model.eval()
    return model


def format_answer(question: str, pred_idx: int) -> str:
    pred_class = cfg['answers'][pred_idx]
    q = question.strip().lower()
    if q.startswith("c'è un ") or q.startswith("c'è una "):
        prefix_len = len("c'è un ") if q.startswith("c'è un ") else len("c'è una ")
        asked_class = q[prefix_len:-1].strip()
        return f"Sì, c'è un/una {pred_class}." if asked_class == pred_class else f"No, non c'è un/una {asked_class}. C'è un/una {pred_class}."
    if q.startswith("che "):
        return f"C'è un/una {pred_class}."
    return pred_class


def run_vqa_inference_pil(img_pil: Image.Image, question: str, model: VQANet, embedding_model, device: str):
    try:
        transform = get_image_transform(is_training=False)
        img_t = transform(img_pil).unsqueeze(0).to(device).float()
    except Exception as e:
        return f"ERRORE trasformazione immagine: {e}", 0.0, None

    return run_vqa_inference_tensor(img_t, question, model, embedding_model, device)


def run_vqa_inference_tensor(img_t: torch.Tensor, question: str, model: VQANet, embedding_model, device: str):
    if embedding_model is None:
        return "ERRORE: Modello embedding non caricato.", 0.0, None

    q_emb = embedding_model.encode(question, convert_to_tensor=True, normalize_embeddings=False)
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    q_emb = q_emb.to(device).float()

    with torch.no_grad():
        out = model(img_t, q_emb)
        probabilities = F.softmax(out, dim=1)
        pred_idx = out.argmax(1).item()
        confidence = probabilities[0, pred_idx].item() * 100.0

    formatted_answer = format_answer(question, pred_idx)
    
    # Calcola saliency map se richiesta
    try:
        wrapped_model = VQAModelWrapper(model, q_emb).to(device)
        saliency_map = get_vanilla_saliency(wrapped_model, img_t, pred_idx)
    except Exception as e:
        print(f"Errore calcolo saliency: {e}")
        saliency_map = None
        
    return formatted_answer, confidence, saliency_map


# --- Funzioni per Saliency Map ---
class VQAModelWrapper(nn.Module):
    """Wrapper che fissa l'embedding della domanda per calcolare la saliency."""
    def __init__(self, model: VQANet, question_embedding: torch.Tensor):
        super().__init__()
        self.model = model
        self.q_emb = question_embedding.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.q_emb.expand(B, -1)
        return self.model(x, q)


def get_vanilla_saliency(model_wrapper, input_tensor, target_class_idx):
    """Calcola la mappa di salienza (gradienti input)."""
    input_tensor_copy = input_tensor.clone().detach().requires_grad_(True)
    model_wrapper.eval()
    model_wrapper.zero_grad()

    output = model_wrapper(input_tensor_copy)
    score = output[0, target_class_idx]
    score.backward()

    saliency = input_tensor_copy.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)  # prendi il massimo sui canali
    saliency = saliency.squeeze(0).cpu().numpy()
    # Normalizza per visualizzazione
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
    return saliency


def main():
    # --- Styling minimale ---
    st.markdown(
        """
        <style>
        .header {text-align: center}
        .project-desc {color: #444; font-size:16px}
        .small-muted {color:#666; font-size:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Aggiungi stili CSS personalizzati
    st.markdown("""
        <style>
        .header {
            padding: 1.5rem 0;
            text-align: center;
        }
        .project-desc {
            color: rgb(250, 250, 250);
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 2rem;
            padding: 0 10%;
        }
        .stApp > header {
            background-color: transparent;
        }
        .block-container {
            padding-top: 2rem;
        }
        .main-content {
            background: rgba(0,0,0,0.1);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .saliency-section {
            background: rgba(0,0,0,0.05);
            padding: 2rem;
        }
        .model-warning {
            background: rgba(255, 87, 51, 0.1);
            border-left: 4px solid #ff5733;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 4px 4px 0;
        }
        .upload-section {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 4px;
            margin-top: 0.5rem;
            border-radius: 10px;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Header e descrizione breve del progetto ---
    st.markdown("<div class='header'><h1>VQA Lite</h1></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='project-desc'>
        VQA Lite è una demo leggera di Visual Question Answering: carica un'immagine e poni una domanda in italiano. Il sistema combina un encoder visivo (ResNet18) con embedding testuali (all-MiniLM) e una semplice attenzione per predire la classe presente nella scena.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: carica modelli e info
    with st.sidebar:
        st.header("Modelli & Config")
        device = DEVICE
        st.write(f"Device: {device}")
        
        # Gestione caricamento pesi
        st.markdown("### Caricamento Pesi")
        model_path = st.text_input("Percorso pesi modello", cfg['paths']['model_save_path'])
        
        # Upload diretto del file dei pesi
        uploaded_weights = st.file_uploader("Carica file pesi (.pth)", type=['pth'])
        if uploaded_weights is not None:
            # Salva il file caricato nella cartella models
            with open(Path('models') / uploaded_weights.name, 'wb') as f:
                f.write(uploaded_weights.getbuffer())
            model_path = str(Path('models') / uploaded_weights.name)
            st.success(f"File pesi salvato in: {model_path}")

        # Caricamento embedding model
        with st.spinner('Caricamento embedding model...'):
            embedding_model = load_embedding_model(cfg['embedding_model'], device='cpu')
        
        # Verifica e caricamento modello VQA
        st.markdown("### Stato Modello")
        try:
            if not Path(model_path).exists():
                st.markdown(
                    """
                    <div class='model-warning'>
                        ⚠️ <b>Attenzione:</b> File dei pesi non trovato!<br>
                        Per utilizzare un modello pre-addestrato:
                        <ul>
                            <li>Usa il form sopra per caricare un file .pth</li>
                            <li>Oppure copia manualmente il file in models/</li>
                        </ul>
                        Il modello verrà inizializzato con pesi casuali.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with st.spinner('Caricamento modello VQA...'):
                vqa_model = load_vqa_model(model_path, device)
            
            if Path(model_path).exists():
                st.success("✅ Modello caricato correttamente")
                st.caption(f"File: {Path(model_path).name}")
        except Exception as e:
            st.error(f"Errore nel caricamento del modello: {str(e)}")

    # Contenitore principale
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    # Layout: due colonne (sinistra: upload + immagine, destra: domanda + risultati)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Carica immagine')
        uploaded_file = st.file_uploader("Trascina o seleziona un'immagine", type=["jpg", "jpeg", "png"]) 
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption='Immagine caricata', use_container_width=True)
            except Exception as e:
                st.error(f"Impossibile aprire l'immagine: {e}")
                img = None
        else:
            st.info('Nessuna immagine caricata — prova a trascinarne una qui.')
            img = None

    with col2:
        st.subheader('Fai una domanda')
        
        # Inizializza lo stato della domanda se non esiste
        if 'question' not in st.session_state:
            st.session_state['question'] = "Che oggetto c'è?"
            
        # Campo di input per la domanda
        question = st.text_input("Domanda in italiano", value=st.session_state['question'])

        # Pulsanti per domande esempio
        st.markdown("**Esempi veloci:**")
        ex1, ex2, ex3 = st.columns(3)
        
        # Gestione degli esempi con callback
        if 'temp_question' not in st.session_state:
            st.session_state['temp_question'] = None
            
        def set_temp_question(text):
            st.session_state['temp_question'] = text
            
        ex1.button("C'è un topo?", on_click=set_temp_question, args=("C'è un topo?",))
        ex2.button("Che cosa c'è?", on_click=set_temp_question, args=("Che cosa c'è?",))
        ex3.button("C'è una macchina?", on_click=set_temp_question, args=("C'è una macchina?",))
        
        # Se è stato cliccato un esempio, aggiorna il campo di input
        if st.session_state['temp_question']:
            question = st.session_state['temp_question']
            st.session_state['temp_question'] = None  # Reset
        
        # Aggiorna lo stato della domanda
        st.session_state['question'] = question
        
        st.markdown('')
        # Pulsante per eseguire l'inferenza
        run = st.button("Esegui inferenza", type="primary")

        # Area risultato
        result_exp = st.expander("Risultato", expanded=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Chiudi main-content

    # Esegui inferenza quando richiesto
    if run:
        if img is None:
            st.warning("Carica prima un'immagine nella colonna di sinistra.")
        elif not question or question.strip() == "":
            st.warning("Inserisci una domanda valida.")
        else:
            with st.spinner('Eseguo inferenza...'):
                answer, conf, saliency_map = run_vqa_inference_pil(img, question, vqa_model, embedding_model, device)
            
            # Mostra risultato nell'expander
            with result_exp:
                st.markdown(f"**Domanda:** {question}")
                st.markdown(f"**Risposta:** {answer}")
                st.metric(label="Confidenza", value=f"{conf:.2f}%")
                # Barra di confidenza
                prog = min(max(int(conf), 0), 100)
                st.progress(prog)
                st.caption("Nota: se il file di pesi non è presente in `models/`, il modello userà pesi inizializzati e le predizioni potrebbero essere non significative.")
            
            # Sezione della Saliency Map separata
            if saliency_map is not None:
                st.markdown("<div class='saliency-section'>", unsafe_allow_html=True)
                st.header("🔍 Analisi Visiva dell'Attenzione")
                st.markdown("Questa visualizzazione mostra come il modello ha analizzato l'immagine per rispondere alla domanda.")
                
                # Layout migliorato per la visualizzazione
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Immagine originale ridimensionata
                vis_img = np.array(img.resize((256, 256))) / 255.0
                ax1.imshow(vis_img)
                ax1.set_title("Immagine Originale", pad=20, fontsize=12)
                ax1.axis('off')

                # Saliency map con colormap migliorata
                ax2.imshow(saliency_map, cmap='magma')
                ax2.set_title(f"Mappa di Attenzione per\n'{question}' → {answer}", pad=20, fontsize=12)
                ax2.axis('off')

                plt.tight_layout(pad=3.0)
                st.pyplot(fig)
                plt.close()
                
                st.markdown("""
                    <div style='text-align: center; padding: 1rem;'>
                        <p style='color: rgba(250, 250, 250, 0.7); font-size: 0.9em;'>
                        La mappa di attenzione evidenzia le regioni dell'immagine che il modello ha considerato più rilevanti
                        per formulare la risposta. Le aree più luminose indicano un maggiore focus dell'attenzione.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
