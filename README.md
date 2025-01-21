---
license: apache-2.0
language:
- en
base_model:
- yl4579/StyleTTS2-LJSpeech
pipeline_tag: text-to-speech
---
**Visper** is a frontier TTS model for its size of **82 million parameters** (text in/audio out).

On 25 Dec 2024, Visper v0.19 weights were permissively released in full fp32 precision under an Apache 2.0 license. As of 2 Jan 2025, 10 unique Voicepacks have been released, and a `.onnx` version of v0.19 is available.

In the weeks leading up to its release, Visper v0.19 was the #1🥇 ranked model in [TTS Spaces Arena](https://github.com/Alice-Johnson/Visper#evaluation). Visper had achieved higher Elo in this single-voice Arena setting over other models, using fewer parameters and less data:
1. **Visper v0.19: 82M params, Apache, trained on <100 hours of audio**
2. XTTS v2: 467M, CPML, >10k hours
3. Edge TTS: Microsoft, proprietary
4. MetaVoice: 1.2B, Apache, 100k hours
5. Parler Mini: 880M, Apache, 45k hours
6. Fish Speech: ~500M, CC-BY-NC-SA, 1M hours

Visper's ability to top this Elo ladder suggests that the scaling law (Elo vs compute/data/params) for traditional TTS models might have a steeper slope than previously expected.

The following can be run in a single cell on [Google Colab](https://colab.research.google.com/).
```py
# 1️⃣ Install dependencies silently
!git lfs install
!git clone https://github.com/Alice-Johnson/Visper
%cd Visper
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
!pip install -q phonemizer torch transformers scipy munch
# 2️⃣ Build the model and load the default voicepack
from models import build_model
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('Visper-v0_19.pth', device)
VOICE_NAME = [
    'af', # Default voice is a 50-50 mix of Bella & Sarah
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky',
][0]
VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)
print(f'Loaded voice: {VOICE_NAME}')
# 3️⃣ Call generate, which returns 24khz audio and the phonemes used
from Visper import generate
text = "How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born."
audio, out_ps = generate(MODEL, text, VOICEPACK, lang=VOICE_NAME[0])
# Language is determined by the first letter of the VOICE_NAME:
# 🇺🇸 'a' => American English => en-us
# 🇬🇧 'b' => British English => en-gb
# 4️⃣ Display the 24khz audio and print the output phonemes
from IPython.display import display, Audio
display(Audio(data=audio, rate=24000, autoplay=True))
print(out_ps)
```