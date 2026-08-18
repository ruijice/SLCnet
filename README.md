# SLCNet - Peer-review supplementary material

This repository contains the peer-review code for the core model described in
the associated IEEE TGRS manuscript.

SLCNet is a standalone single-stage detector for single-category
horizontal bounding-box (HBB) ship detection in coastal synthetic aperture
radar (SAR) imagery. The implementation contains the core hierarchical feature
extractor, soft coastal guidance, local context aggregation, small-object
feature pathway, decoupled HBB prediction head, native decoding, and basic
COCO data/evaluation interfaces. No external detector framework is required.

## Scope of this Peer-Review Material

This repository provides the core implementation necessary for reviewers to
inspect and run SLCNet. It is intentionally not a self-contained reproduction
archive. Dataset images, checkpoints, exact training parameters, complete
experiment configurations, split manifests, and detailed reproduction records
are withheld during peer review. Reproducing the manuscript results therefore
requires controlled preprocessing and additional author-provided material; the
files in this repository alone must not be treated as a complete reproduction
package.

The peer-review material does not contain dataset images or generated labels.
The intended evaluation scope is `640 x 640` letterboxed SAR HBB detection on
HRSID and SSDD. Oriented bounding boxes, cross-sensor fusion, and other tasks
are outside the scope of the manuscript.

## Main manuscript results

The values below are the principal results reported in the associated TGRS
manuscript. They are reported manuscript values and are not a new evaluation
performed from this incomplete peer-review material.

| Dataset |           mAP@0.5 |      mAP@0.5:0.95 |                F1 |            Recall |           FPS |
| ------- | ----------------: | ----------------: | ----------------: | ----------------: | ------------: |
| HRSID   | 0.9425 +/- 0.0038 | 0.7120 +/- 0.0052 | 0.9125 +/- 0.0025 | 0.8805 +/- 0.0045 |  85.9 +/- 2.5 |
| SSDD    | 0.9802 +/- 0.0015 | 0.6935 +/- 0.0012 | 0.9485 +/- 0.0010 | 0.9430 +/- 0.0025 | 110.6 +/- 3.2 |

The manuscript reports SLCNet as an accuracy-recall-throughput operating point
for coastal SAR HBB detection. These numbers must be interpreted only under
the paper's stated datasets, splits, preprocessing, hardware, and evaluation
protocol.

## Confidentiality and License

This repository is provided strictly for the IEEE TGRS peer-review process. The
code, architectures, and results contained herein are confidential and
proprietary prior to publication. No license is granted for copying,
modifying, reproducing, distributing, or utilizing this code for any academic
or commercial purposes at this stage. A formal open-source license will be
assigned only after the manuscript is officially accepted and published. Do
not distribute or cite this repository.

Copyright (c) 2026 the authors. All rights reserved during peer review.
