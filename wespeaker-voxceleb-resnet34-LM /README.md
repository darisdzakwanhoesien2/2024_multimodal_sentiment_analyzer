---
license: apache-2.0
---
# WeSpeaker ResNet34 speaker embedding

This is a copy of `voxceleb_resnet34_LM.onnx` speaker embedding model taken from [ChengDong Liang's repository](https://huggingface.co/chengdongliang/wespeaker).

```
Copyright 2022 ChengDong Liang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## How to use with [pyannote.audio](https://github.com/pyannote/pyannote-audio)

```python
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
get_embedding = PretrainedSpeakerEmbedding("hbredin/wespeaker-voxceleb-resnet34-LM")

assert waveforms.ndim == 3
batch_size, num_channels, num_samples = waveforms.shape
assert num_channels == 1

embeddings = get_embedding(waveforms)
assert embeddings.ndim == 2
assert embeddings.shape[0] == batch_size
```
