# Meta-Pseudo-Label-for-EHR-data

**Project Introduction**

Electronic health records are unstructured data without a set structure. This textual information facilitates the expression of statistical analysis and other studies creating obstacles. Named Entity Recognition is the most fundamental step in analyzing medical knowledge in electronic health records. Boosting a pre-trained model based on a specific corpus is time-consuming, so improving the model by fine-tuning is preferred. This project uses a novel semi-supervised learning knowledge distillation method, Meta Pseudo Label, to fine-tune a Named Entity model for electronic health records data. Finally, map various ways of expressing the concepts in clinical patient notes by medical students to clinical concepts from standard rules.

**Meta Pseudo Label**

Pseudo-labeling, a method of fine-tuning, is a semi-supervised learning technique that can be used to improve the performance of machine learning models. It involves using a trained model, denoted as the Teacher model, to make predictions on a set of unlabeled data and then using the predicted labels as the "pseudo labels" for the unlabeled data. The labeled and pseudo-labeled data are combined to train a new student model. Pseudo-labeling can be particularly useful when relatively few labeled examples are available, as it allows the model to learn from a larger dataset and potentially improve its performance.

Despite the superior performance of the pseudo-labeling method, it also has a significant drawback. If the pseudo-labeling is inaccurate, the student model has to learn from the incorrect data. As a result, the final trained student model may not be much better than the teacher model. This drawback is also known as the confirmation bias problem of pseudo-labeling. To address this issue, the Teacher model needs to correct for bias through the effect of its pseudo-labels on the Student model, which is exactly Meta Pseudo model (MPL).

For more details: [Poster](https://github.com/FayeFaye-wong/Meta-Pseudo-Label-for-EHR-data/blob/main/poster.pdf)
