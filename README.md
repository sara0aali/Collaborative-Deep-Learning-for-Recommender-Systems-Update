# Collaborative-Deep-Learning-for-Recommender-Systems/Update

The hybrid model combining stacked denoising autoencoder (SDAE) with matrix factorization (MF) is applied, to predict the customer purchase behavior in the future month according to the purchase history and user information in the [Santander dataset](https://www.kaggle.com/c/santander-product-recommendation). A blog post for some follow up discussions can be found [here](https://xiaoouzhang.github.io/collaborative/).

This work is contributed by [Sampath Chanda](https://www.linkedin.com/in/sampathchanda/), [Suyin Wang](https://www.linkedin.com/in/suyin-wang-3934b543/) and [Xiaoou Zhang](https://www.linkedin.com/in/xiaoou-zhang-a9559211a/).


Based on the original project, The user information is generated in "matrix_factorization.ipynb" and the rating matrix is ​​generated in "rating_matrix.py". The original code of the SDAE-MF hybrid model can be found in "mf_auto_mono.py" for the single-layer SDAE and "mf_auto.py" for the three-layer hidden SDAE.

The updates to the original project are as follows: In "mf_auto_mono_v2.py" and "mf_auto_v2.py", the speed of matrix factorization is improved by exploiting the dispersion of the rating matrix.

You can view the original project via the link:(https://github.com/xiaoouzhang/Collaborative-Deep-Learning-for-Recommender-Systems.git)

In this project, we focused on improving the original implementation of the Collaborative Deep Learning model by introducing several enhancements and optimizations. These changes address both computational efficiency and the explainability of recommendations.

Enhancements and Modifications:

1. Matrix Factorization (matrix_factorization.ipynb):
   - Original: Generated user information for the model.  
   - Enhancements: Streamlined the process to better handle sparsity and significantly improved the efficiency of data handling.

2. Rating Matrix Generation (rating_matrix.py): 
   - Original: Created the rating matrix used for training the model.  
   - Enhancements: Leveraged sparse matrix operations to enhance the efficiency of matrix creation.

3. Hybrid Model Implementation:
   - Single-Hidden-Layer SDAE (mf_auto_mono.py):
     - Original: Developed the hybrid model with a single hidden layer.  
     - Enhancements: Integrated L2 regularization and optimized gradient descent, leading to improved model training and prediction accuracy.  
   - Three-Hidden-Layer SDAE (mf_auto.py):
     - Original: Extended the model to include three hidden layers for deeper feature extraction.  
     - Enhancements: Applied similar improvements, with adjustments to manage the increased complexity.

4. Testing and Explainability (testing.py): 
   - Original: Generated predictions and evaluated percentile ranking for recommendations.  
  Enhancements:  
     - Introduced an explainability function(`explain_recommendation`) to break down and analyze the contribution of each latent dimension to the recommendation score.  
     - Added functionality to save detailed explanations in a file using the `save_explanations_to_file` method. This enables better transparency and reporting of recommendation results.

These enhancements collectively improve the model's performance and provide deeper insights into the recommendation process, making the system more efficient and user-friendly.

Ensure that all dependencies are installed as specified in the project's requirements file.

