# streamlit_app_joanna

Run everything from inside this folder:

    cd streamlit_app_joanna

Install dependencies (repo-wide):

    pip install -r ../requirements.txt

Train the baseline hardhat/no-hardhat classifier:

    python train_baseline_classifier.py

Train YOLOv8n from scratch (no pretrained weights):

    python train_yolo_scratch.py

Run the app:

    streamlit run app.py

Settings (epochs, batch size, dataset path, ...) are constants at the top of each
train_*.py file — edit the file directly, no CLI flags.

Trained models save locally to `models/` (gitignored) — for now, each teammate
either trains their own or grabs a trained file from someone else (Slack/Drive)
and drops it in that folder. Not solved yet: if this app gets deployed anywhere
other than a laptop, it'll need the weights from somewhere fetchable instead —
a shared Drive folder is the lowest-effort option given we're already using
Colab/Kaggle for GPU training.
