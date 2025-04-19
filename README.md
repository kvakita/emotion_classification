## Multimodal Emotion Classification in Dialogues

This project builds a multimodal neural network for emotion recognition in dialogues using the MELD dataset. It uses textual information (via BERT) and includes functionality for automatic dataset download.

### Structure:
- `data/`: MELD dataset will be automatically downloaded here
- `models/`: Trained model weights
- `notebooks/`: Optional notebooks for exploration
- `main.py`: Core training + evaluation script

### How to Run:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python main.py`

The script will automatically download the MELD dataset if it's not found locally.

### Future Improvements:
- Add audio & visual modality integration
- Use attention mechanisms for modality fusion
- Deploy as a service (REST API)
