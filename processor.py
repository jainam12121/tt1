import onnxruntime
import soundfile as sf
import yaml
from ttstokenizer import TTSTokenizer

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
# model = onnxruntime.InferenceSession(
#     "./model.onnx",
#     providers=["CPUExecutionProvider"]
# )

# Create tokenizer
tokenizer = TTSTokenizer(yaml_config_dict["token"]["list"])

def pre_process(text):
    """Pre-processes the input text by tokenizing it."""
    #print("Tokenizing input text...")
    tokenized_input = tokenizer(text)
    #print("Tokenized input:", tokenized_input)
    return tokenized_input

def post_process(wav):
    output_file="out.wav"
    """Processes the model output and saves it as a .wav file."""
    #print("Saving audio output...")
    sf.write(output_file, wav, 22050)
    #print(f"Audio saved to {output_file}")
    return wav  # Return the processed audio data
    # print(f"Audio saved to {output_file}")

# Main execution
# try:
#     # Define input text
#     input_text = "hi i am jainam"

#     # Tokenize inputs
#     tokenized_inputs = pre_process(input_text)

#     # Generate speech
#     outputs = model.run(None, {"text": tokenized_inputs})

#     # Save the generated audio
#     audio_data = outputs[0]  # Assuming the output is in the first position
#     post_process(audio_data)

#     print("Audio processing completed successfully.")

# except Exception as e:
#     print(f"An error occurred during audio generation: {str(e)}")
# finally:
#     print("Script execution finished.")
