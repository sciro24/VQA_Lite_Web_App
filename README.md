## VQA Lite — Lightweight Visual Question Answering (VQA) demo

Questo repository contiene una semplice demo di Visual Question Answering (VQA) costruita con Streamlit, PyTorch e un modello di embedding testuale (SentenceTransformers). L'app permette di caricare un'immagine (o selezionarne una tra quelle già fornite), porre una domanda in italiano e ottenere una risposta testuale con una misura di confidenza. È anche possibile visualizzare una mappa di salienza (attenzione) che mostra le regioni dell'immagine utilizzate dal modello.

Repository structure
--------------------

- `app.py` - applicazione Streamlit principale.
- `requirements.txt` - dipendenze Python.
- `models/` - cartella per i pesi del modello VQA (es. `vqa_model_best.pth`).
- `test_images/` - cartella dove inserire immagini di test già fornite (puoi aggiungerne di tue). Contiene un `.gitkeep` di esempio.
- `data/` - eventuali dataset (utili per training/esperimenti locali).

Obiettivo
---------

Fornire una demo locale e leggera per sperimentare una pipeline VQA: encoder visivo (ResNet18) + embedding testuale (all-MiniLM) + meccanismo di attenzione semplice. L'interfaccia consente di caricare un'immagine o scegliere una di quelle presenti in `test_images/`, quindi porre domande in italiano.

Prerequisiti
-------------

- macOS / Linux / Windows con Python 3.8+ (i comandi qui sotto assumono macOS / zsh).
- GPU non obbligatoria: il codice rileva automaticamente CUDA se disponibile. Su CPU l'esecuzione è più lenta.

Installazione (locale)
----------------------

1. Clona il repository (se non l'hai già fatto):

```bash
git clone https://github.com/sciro24/VQA_Lite_Web_App.git
cd VQA_Lite_Web_App
```

2. Crea e attiva un ambiente virtuale (consigliato):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Nota: il file `requirements.txt` dovrebbe includere almeno `streamlit`, `torch`, `torchvision`, `sentence-transformers`, `pillow`, `matplotlib`, e altre librerie usate nel progetto. Se riscontri errori, verifica la versione di PyTorch compatibile con la tua piattaforma (es. CPU vs CUDA).

Modelli e pesi
--------------

- Il codice cerca automaticamente i pesi del modello in `models/vqa_model_best.pth` (configurabile tramite la sidebar nell'app). Se non hai un file di pesi, puoi caricarlo dall'interfaccia sidebar oppure copiarlo manualmente nella cartella `models/`.
- Se non è presente alcun file di pesi, il modello verrà inizializzato casualmente e le predizioni non saranno significative. Per usare predizioni sensate, fornisci i pesi addestrati.

Usare l'app (locale)
--------------------

1. Avvia l'app Streamlit:

```bash
streamlit run app.py
```

2. Nel pannello laterale (`sidebar`) puoi:
- vedere su quale device viene eseguito il codice (CPU/GPU),
- impostare il percorso del file dei pesi del modello (o caricarne uno nuovo),
- il codice caricherà automaticamente il modello VQA e l'encoder di embedding (SentenceTransformers).

3. Nella pagina principale:
- Colonna sinistra: puoi caricare la tua immagine o selezionare un'immagine di test da `test_images/` (se presenti). Se carichi un'immagine, questa ha priorità su una immagine di test selezionata.
- Colonna destra: inserisci la domanda in italiano e premi "Esegui inferenza" per ottenere la risposta, una confidenza e (se disponibile) la mappa di salienza.

Aggiungere immagini di test
--------------------------

Per preparare immagini di test da usare nell'interfaccia senza caricarle ogni volta, copia file `.jpg`, `.jpeg` o `.png` dentro la cartella `test_images/`. L'app mostra automaticamente le immagini disponibili in quella cartella e permette di selezionarle.

Esempio:

```bash
cp ~/Downloads/mia_immagine.jpg test_images/
```

Consigli e note tecniche
-----------------------

- Dimensione input: l'immagine viene ridimensionata internamente (32x32 nel flusso attuale) per adattarsi al modello leggero usato nella demo. Questo è intenzionale per mantenere la demo snella, ma limita la qualità dell'attenzione e delle predizioni per immagini complesse.
- Embedding domande: usiamo `sentence-transformers` (configurabile in `cfg['embedding_model']` in `app.py`). Alcuni modelli di embedding possono richiedere download iniziali.
- Se usi una GPU assicurati che la versione di PyTorch installata supporti CUDA per la tua GPU.

Debug e troubleshooting
-----------------------

- Errore di import / versione PyTorch:
	- Controlla che la versione di Python sia corretta e che `pip install -r requirements.txt` non abbia restituito errori.
	- Per installare PyTorch con supporto CUDA visita https://pytorch.org/ e segui le istruzioni per la tua piattaforma.
- L'app non trova il file di pesi: copia il file `.pth` in `models/` o usa il caricatore nella sidebar.
- Immagini non visibili: verifica che i file nella cartella `test_images/` abbiano estensione `.jpg`, `.jpeg` o `.png` e permessi di lettura.

Contributi e sviluppo
----------------------

Se vuoi contribuire:
- Apri un issue per proporre miglioramenti o segnalare bug.
- Fai una fork, crea un branch, ed invia una pull request per cambi significativi.

Possibili miglioramenti futuri
-----------------------------

- Supporto per immagini ad alta risoluzione e rete visiva più potente.
- Interfaccia per aggiungere/gestire le immagini di test da GUI (upload a server o drag-and-drop nella cartella `test_images`).
- Aggiungere test automatici e un piccolo set di immagini di esempio con relative domande/risposte per demo ripetibili.

Licenza
-------

Questo progetto è rilasciato sotto licenza MIT — vedi il file `LICENSE` (se presente) o aggiungi una licenza a tuo piacimento.

Contatti
--------

Per domande o richieste: apri un issue su GitHub o contatta il maintainer del repository.

Buon divertimento con VQA Lite!

# VQA_Lite_Web_App