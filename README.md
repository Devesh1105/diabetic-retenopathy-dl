Diabetic Retinopathy Detection 

Diabetic Retinopathy (DR) is a diabetes-related eye disease and one of the leading causes of preventable blindness. Early detection plays a crucial role in reducing vision loss; however, manual screening of retinal fundus images is time-consuming, resource-intensive, and often limited by the availability of trained specialists.

This project presents an automated deep learning–based approach for the classification of Diabetic Retinopathy severity from retinal fundus images. The dataset used in this study is sourced from Roboflow Universe:
https://universe.roboflow.com/drproject/diabetic-retenopathy

Due to limited computational resources, three different model variations were trained using different preprocessing and augmentation strategies, and the best-performing model was selected based on validation performance.

The proposed framework utilizes an Attention U-Net architecture with a ResNet50 encoder for robust feature extraction. To address the severe class imbalance commonly present in DR datasets, focal loss was employed to emphasize hard-to-classify samples. In addition, class-specific data augmentation was applied using the Albumentations library, with stronger transformations for underrepresented classes such as Severe and Proliferative DR.

Images were resized to 224×224 pixels and normalized before training. 
The model was trained with a batch size of 16 for up to 30 epochs, with early stopping applied at 28 epochs.
Performance evaluation was conducted using accuracy, balanced accuracy, and Cohen’s Kappa to ensure fair assessment across all severity classes.

The final model(initial model 1) achieved a test accuracy of 78.54%, a balanced accuracy of 67.85%, and a Cohen’s Kappa score of 0.6797, demonstrating improved detection of minority DR stages compared to baseline models. These results highlight the effectiveness of focal loss and targeted data augmentation in handling imbalanced medical datasets.

Overall, this project demonstrates a scalable and efficient framework for automated DR screening under hardware constraints. Future work may explore ensemble learning, explainable AI techniques, and real-time deployment in clinical or telemedicine settings.
