# CTR Prediction and Synthetic Data Generation

Data analytics using predictive AI algorithms and synthesizing with suitable evaluations for ad click-through rate (CTR) prediction. This repository includes code for a **flow-matching-based model** to generate high-fidelity synthetic data, balancing utility and privacy.

## Dataset

This project uses the Kaggle dataset from the CTR Prediction - 2022 DIGIX Global AI Challenge:  
[Dataset Link](https://www.kaggle.com/datasets/xiaojiu1414/digix-global-ai-challenge/data)

## Features

- **Predictive Modeling**:
  - Tested machine learning models including XGBoost, Logistic Regression, Random Forest, MLP, and Transformer.
  - Evaluated using accuracy, ROC-AUC, precision, recall, and F1-Score.

- **Synthetic Data Generation**:
  - Flow-based generative models using **Conditional Flow Matching (CFM)**.
  - Synthetic data maintains high fidelity to the real dataset while preserving user privacy.

- **Evaluation Metrics**:
  - Privacy: Nearest Neighbor Distance Ratio (NNDR), Distance to Closest Record (DCR).
  - Fidelity: KL Divergence, Earth Mover's Distance (EMD).

## Implementation

- Flow-based generative modeling implemented with [TorchCFM](https://github.com/atong01/conditional-flow-matching).
- Preprocessing pipelines include normalization, noise duplication, and class balancing.
- XGBoost used as a primary vector field approximator for CTR predictions.

## Results

- Incorporating synthetic data improved F1-Score and precision while maintaining high privacy.
- Synthetic datasets demonstrated strong alignment with real data using statistical fidelity metrics (e.g., T-SNE, KDE plots).

## Credits

This repo uses the TorchCFM implementation from:  
[TorchCFM GitHub](https://github.com/atong01/conditional-flow-matching)  

This project was developed with the assistance of **ChatGPT by OpenAI**, which provided guidance on debugging, optimization, and documentation tasks.

## Future Directions

- Benchmarking against generative models like GANs and VAEs.  
- Scaling synthetic data generation for larger datasets and diverse domains.  
- Applying the model to use cases like recommendation systems and fraud detection.

