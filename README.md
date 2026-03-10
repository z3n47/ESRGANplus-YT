# ESRGANplus-YT – Single Image Super-Resolution using ESRGAN+

Implementation and experimentation with **Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)** for the task of **Single Image Super-Resolution (SISR)**.

This project explores deep learning techniques to reconstruct **high-resolution (HR) images from low-resolution (LR) inputs**, using improved GAN-based architectures derived from **SRGAN and ESRGAN**.

The project was developed as part of my **MSc in Advanced Computer Science** research in **Computer Vision and Deep Learning**.

---

## Project Overview

Image Super-Resolution (SR) is a computer vision task aimed at enhancing the spatial resolution of images. The goal is to generate a **high-resolution image from a low-resolution input while preserving fine details and textures**.

Traditional interpolation methods (bicubic, bilinear) often produce blurred images. Deep learning approaches such as **Generative Adversarial Networks (GANs)** significantly improve perceptual quality.

This project implements **ESRGAN+**, which builds upon SRGAN with architectural improvements to generate more realistic high-resolution images.

Key improvements include:

- Residual-in-Residual Dense Blocks (RRDB)
- Relativistic Average Discriminator
- Improved perceptual loss functions

These innovations allow the model to reconstruct detailed textures and sharper edges.

---

## Key Features

- Implementation of **ESRGAN-based image super-resolution**
- Exploration of **deep learning GAN architectures for computer vision**
- GPU experimentation using **Google Colab**
- Support for **pre-trained models for inference**
- Image reconstruction using **deep neural networks**

---

## Technologies Used

- Python
- PyTorch / TensorFlow
- OpenCV
- NumPy
- Google Colab
- VSCode


References

Wang et al., 2018 – ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
https://arxiv.org/abs/1809.00219

Ledig et al., 2017 – Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
https://arxiv.org/abs/1609.04802
