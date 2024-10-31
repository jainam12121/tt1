import onnxruntime
import soundfile as sf
import yaml
import logging
from ttstokenizer import TTSTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded YAML configuration
yaml_config = """
normalize:
  eps: 1.0e-20
  norm_means: true
  norm_vars: true
  stats_file: imdanboy/ljspeech_tts_train_jets_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave/feats_stats.npz
  type: gmvn
  use_normalize: true
text_cleaner:
  cleaner_types: tacotron
token:
  list:
  - <blank>
  - <unk>
  - AH0
  - N
  - T
  - D
  - S
  - R
  - L
  - DH
  - K
  - Z
  - IH1
  - IH0
  - M
  - EH1
  - W
  - P
  - AE1
  - AH1
  - V
  - ER0
  - F
  - ','
  - AA1
  - B
  - HH
  - IY1
  - UW1
  - IY0
  - AO1
  - EY1
  - AY1
  - .
  - OW1
  - SH
  - NG
  - G
  - ER1
  - CH
  - JH
  - Y
  - AW1
  - TH
  - UH1
  - EH2
  - OW0
  - EY2
  - AO0
  - IH2
  - AE2
  - AY2
  - AA2
  - UW0
  - EH0
  - OY1
  - EY0
  - AO2
  - ZH
  - OW2
  - AE0
  - UW2
  - AH2
  - AY0
  - IY2
  - AW2
  - AA0
  - ''''
  - ER2
  - UH2
  - '?'
  - OY2
  - '!'
  - AW0
  - UH0
  - OY0
  - ..
  - <sos/eos>
tokenizer:
  g2p_type: g2p_en_no_space
  token_type: phn
tts_model:
  model_path: imdanboy/ljspeech_tts_train_jets_raw_phn_tacotron_g2p_en_no_space_train.total_count.ave/full/jets.onnx
  model_type: JETS
vocoder:
  vocoder_type: not_used
"""

# Parse YAML configuration
yaml_config_dict = yaml.safe_load(yaml_config)

# Create model
model = onnxruntime.InferenceSession(
    "./model.onnx",
    providers=["CPUExecutionProvider"]
)

# Create tokenizer
tokenizer = TTSTokenizer(yaml_config_dict["token"]["list"])

def pre_process(text):
    """Pre-processes the input text."""
    logger.info(f"Input text: {text}")
    try:
        tokenized_input = tokenizer(text)
        logger.info(f"Tokenized input: {tokenized_input}")
        return tokenized_input
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        raise

def post_process(wav):
    """Post-processes the generated audio."""
    logger.info(f"Processing audio file: {wav}")
    try:
        audio_data = sf.read(wav)[0]  # Extract only the audio data
        logger.info(f"Audio duration: {len(audio_data) / 22050:.2f} seconds")
        return audio_data
    except Exception as e:
        logger.error(f"Error reading audio file: {str(e)}")
        raise

# Main execution
try:
    # Tokenize inputs
    input_text = "Say something here"
    tokenized_inputs = pre_process(input_text)

    # Generate speech
    outputs = model.run(None, {"text": tokenized_inputs})
    
    # Write to file
    output_wav = post_process("out1.wav")

    # Save processed audio
    sf.write("out.wav", output_wav, 22050)
    logger.info("Audio processing completed successfully.")

except Exception as e:
    logger.exception(f"An error occurred during audio generation: {str(e)}")
finally:
    logger.info("Script execution finished.")
