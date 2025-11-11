# Multimodal-Classifier-for-Disaster-Images

Multimodal Disaster Classification (EfficientNet + BERT)

This project classifies disaster-related image–tweet pairs into six categories using a multimodal deep learning model that combines EfficientNet for image features and BERT for text features through early (feature-level) fusion.

🧠 Overview
Goal: Detect disaster type (e.g., fire, flood, human damage) using both visual and textual information.
Approach: Early fusion of image and text embeddings to improve accuracy and generalization.
Modalities Used:
Image → EfficientNet-B3 (feature extraction)
Text → BERT-base-uncased (contextual understanding)

📂 Dataset
Input: paired images and tweets related to disasters.
Total 6 classes:
Non-Damage
Damaged Infrastructure
Damaged Nature
Fires
Flood
Human Damage
Dataset is imbalanced, with Non-Damage as the majority class.

⚙️ Model Architecture
Image Encoder: EfficientNet extracts 512-dimensional visual features.
Text Encoder: BERT generates 512-dimensional contextual embeddings.
Fusion: Concatenation of both features (1024-D) → Linear classifier.
Output: 6 disaster category predictions.

Key Features
Early fusion for joint learning of visual and textual cues.
Data augmentation for robustness.
Dropout (0.2) to prevent overfitting.
Handles class imbalance effectively.
Reproducible results (seed = 50).

📈 Evaluation Metrics
Accuracy: Overall correctness.
Precision & Recall: Class-specific reliability and sensitivity.
F1-Score: Balance between precision and recall (weighted for imbalance).
