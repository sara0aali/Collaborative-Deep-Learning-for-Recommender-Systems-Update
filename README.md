In the **Collaborative Deep Learning for Recommender Systems** project, a hybrid model combining the **Stacked Denoising Autoencoder** (SDAE) with **Matrix Factorization** (MF) is applied to predict customer purchase behavior for the upcoming month based on purchase history and user information in the Santander dataset.

This project was developed by **Sampath Chanda**, **Suyin Wang**, and **Xiaoou Zhang**.

- User information is generated in **"matrix_factorization.ipynb"**, and the rating matrix is produced in **"rating_matrix.py"**.
- The main code for the SDAE-MF hybrid model is found in **"mf_auto_mono.py"** (for a single-hidden-layer SDAE) and **"mf_auto.py"** (for a three-hidden-layer SDAE).

### Updates and Optimizations
In **"mf_auto_mono_v2.py"** and **"mf_auto_v2.py"**, the performance of matrix factorization has been enhanced by leveraging the sparsity of the rating matrix, alongside additional optimizations in **L2 Regularization** and **Gradient Descent**, which further improve the accuracy and efficiency of the model.
