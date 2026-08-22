# Data Availability

The datasets analyzed in this study are publicly available for academic
research under their respective access policies.

## Celeb-DF-v1

Celeb-DF-v1 is used as the source-domain dataset for training, validation, and
in-domain evaluation. The fixed split used in this project contains 1,203
videos before face-extraction filtering:

```text
train: 971 videos
val:   132 videos
test:  100 videos
```

The dataset contains 408 real videos and 795 DeepFake videos in total. For
compression-robust evaluation, cached face clips are prepared under:

```text
crf_src
crf0
crf23
crf40
```

## Celeb-DF-v2

Celeb-DF-v2 is used as an external cross-dataset evaluation set. The public
split included in this repository contains a balanced subset:

```text
300 real videos
300 fake videos
```

Celeb-DF-v2 is not used for training the final GRFR model.

## Redistributed Files

This repository includes:

```text
splits.json
splits_celebdfv2_300.json
preprocessing scripts
training scripts
evaluation scripts
selected result files
```

This repository does not redistribute:

```text
raw videos
face-cache images
model checkpoints
local manuscript drafts
```

Derived face-cache files can be reproduced with `data_pipeline/preprocess_faces.py`
after obtaining access to the original datasets.
