import torch
from speechbrain.inference.interfaces import foreign_class
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

class CustomEncoderWav2vec2Classifier(torch.nn.Module):
    """
    A custom class that combines a HuggingFace Wav2Vec2 model with a downstream
    classifier for tasks like emotion recognition.
    """
    def __init__(self, source, **kwargs):
        super().__init__()
        # Instantiate the Wav2Vec2 model from the HuggingFace source
        self.encoder = HuggingFaceWav2Vec2(source, **kwargs)
        # The pre-trained classifier is also loaded via foreign_class
        self.classifier = foreign_class(source=source, pymodule_file="model.py", classname="Classifier")

    def forward(self, wav):
        """
        Forward pass that takes a waveform and returns the classification result.
        """
        # Get embeddings from the encoder
        emb = self.encoder(wav)
        # Pass embeddings to the classifier
        return self.classifier(emb)

    def classify_file(self, file_path):
        """
        Convenience method to classify a single audio file.
        """
        # This is a method that SpeechBrain's interface expects to exist.
        # It handles loading the audio and making the prediction.
        from speechbrain.dataio.dataio import read_audio
        wav = read_audio(file_path)
        wav = wav.unsqueeze(0)
        out = self.forward(wav)
        out = out.squeeze(0)
        score, index = torch.max(out, dim=-1)
        # The model's labels are part of the classifier's output_neurons
        text_lab = self.classifier.hparams.output_neurons[index]
        return out, score, index, text_lab
