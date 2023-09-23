UNet
----
Implementation of UNet, applying basic UNet and transfer learning.

Mainly contain three files, `model_part.py`, `model.py`, `train.py`.

1. `model_part.py`, contains function for **double convolution layer**, **down sampling**, and **up sampling**.

2. `model.py` is the main model, which build the basic UNet model from the origin paper.

3. `train.py` helps us to train the model and save model.
