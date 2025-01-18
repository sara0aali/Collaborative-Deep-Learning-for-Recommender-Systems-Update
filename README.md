In this project, we enhance the original Collaborative Deep Learning for Recommender Systems by introducing several optimizations to the hybrid model that combines the Stacked Denoising Autoencoder (SDAE) with Matrix Factorization (MF). This model aims to predict customer purchase behavior for the upcoming month based on historical purchase data and user information from the Santander dataset.

Project Contributors

Sampath Chanda
Suyin Wang
Xiaoou Zhang

Original Project
The original implementation of this project can be found on GitHub: Collaborative-Deep-Learning-for-Recommender-Systems.

Enhancements and Modifications

Matrix Factorization (matrix_factorization.ipynb):

Original: Generated user information for the model.
Enhancements: Optimized the generation process to better handle sparsity and efficiency in data handling.

Rating Matrix Generation (rating_matrix.py):

Original: Created the rating matrix used for training the model.
Enhancements: Improved the efficiency of matrix creation by leveraging sparse matrix operations.

Hybrid Model Implementation:

Single-Hidden-Layer SDAE (mf_auto_mono.py):
Original: Implemented the hybrid model with a single hidden layer.
Enhancements: Introduced L2 regularization and gradient descent optimizations to improve model training and prediction accuracy.

Three-Hidden-Layer SDAE (mf_auto.py):
Original: Extended the model with three hidden layers for deeper feature extraction.
Enhancements: Same as above, with adjustments to accommodate the increased complexity.

Testing and Explainability (testing.py):

Original: Calculated model predictions and evaluated percentile ranking.
Enhancements: Added an explainability function (`explain_recommendation`) to analyze and display the contributions of each latent dimension to the recommendation score. This function provides detailed explanations for why a specific item is recommended to a user. Additionally, explanations can be saved to a file using the `save_explanations_to_file` function, enabling easier reporting and analysis.

Ensure that all dependencies are installed as specified in the project's requirements file.
