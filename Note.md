- Data splits:
  - Ensure the 30 listed test IDs in the README are not part of your training data
  - Ensure that the remaining training dataset to create a test and validation scheme, either K-fold validation or single train and validation split (the real images are not in great volumes).

- Data:
  - Focus on the photos (these are the actual images which would simulate a production environment), the remaining are there as augmentation possibilities
  - Remember to add common image augmentations (flip, rotate, crop etc)
  - Finding the right combination to maximize prediction performance on validation data

- Preprocessing:
  - Grayscaling
  - CLAHE or similar normalization techniques

- Architectures:
  - Unet, Unet++, Unet+++ (Unet can provide a good baseline)
  - DeepLabV3+
  - Pix2Pix GAN
  - Gated SCNN
  - transformer

- Loss functions:
  - Categorical cross entropy
  - DICE loss
  - Weighted combined loss (eg., DICE*w1 + CCE*w2)
  - Other categorical loss functions
