import numpy as np
import soundfile as sf
from ttstokenizer import TTSTokenizer
import nltk

nltk.download('averaged_perceptron_tagger_eng')

# Hardcoded configuration
config = {
    "normalize": {
        "use_normalize": False
    },
    "text_cleaner": {
        "cleaner_types": ["tacotron"]
    },
    "token": {
        "list": [
            "<blank>", "<unk>", "AH0", "T", "N", "S", "R", "IH1", "D", "L", ".", "Z", "DH", "K", "W", "M", "AE1", 
            "EH1", "AA1", "IH0", "IY1", "AH1", "B", "P", "V", "ER0", "F", "HH", "AY1", "EY1", "UW1", "IY0", "AO1", 
            "OW1", "G", ",", "NG", "SH", "Y", "JH", "AW1", "UH1", "TH", "ER1", "CH", "?", "OW0", "OW2", "EH2", 
            "EY2", "UW0", "IH2", "OY1", "AY2", "ZH", "AW2", "EH0", "IY2", "AA2", "AE0", "AH2", "AE2", "AO0", "AO2", 
            "AY0", "UW2", "UH2", "AA0", "AW0", "EY0", "!", "UH0", "ER2", "OY2", "'", "OY0", "<sos/eos>"
        ]
    },
    "tokenizer": {
        "g2p_type": "g2p_en_no_space",
        "token_type": "phn"
    },
    "tts_model": {
        "model_path": "espnet/kan-bayashi_vctk_tts_train_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave/full/vits.onnx",
        "model_type": "VITS"
    },
    "vocoder": {
        "vocoder_type": "not_used"
    }
}

# Initialize the ONNX model


# Create tokenizer
tokenizer = TTSTokenizer(config["token"]["list"])

def pre_process(text):
    """Tokenizes input text."""
    tokenized_input = tokenizer(text)
    return tokenized_input

def post_process(output):
    """Processes model output and saves it as a .wav file."""
    audio_data = output[0]
    output_file = "out.wav"
    sf.write(output_file, audio_data, 22050)
    return output_file




