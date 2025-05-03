import numpy as np
from sklearn.metrics import accuracy_score



def moRF_LeRF_SHAP(
    model,
    y_true,
    explaination,
    metric=accuracy_score,
    steps=5,
    mask_top= True
):

    n_samples = len(explaination)
    
    # We'll remove fractions from 0 to 1.0 in 'steps' increments:
    fractions_removed = np.linspace(0, 1, steps + 1)
    performances = []

    # Convert y_true to a numpy array for convenience
    y_true = np.array(y_true)

    for frac in fractions_removed:
        # Build a masked version of the entire dataset, one sample at a time
        X_masked_list = []
        for i in range(n_samples):
            # Original sample's features and shap values
                # shape: (n_i, ...)

            # Number of features to remove
            x_i = explaination[i].data
            n_i = len(x_i)
            n_remove = int(frac * n_i)
            # Sort features by absolute shap magnitude (descending)
            sorted_idx = np.argsort(np.abs(explaination[i].values), axis=0)
            if mask_top:
                sorted_idx = sorted_idx[::-1]


            top_indices = sorted_idx[:n_remove]
            # Mask those features
            masked_sample = [word for j, word in enumerate(x_i) if j not in top_indices]
            #print(len(masked_sample))
            masked_sentence = " ".join(masked_sample)

            X_masked_list.append(masked_sentence)

        y_pred = []
        for masked_sample in X_masked_list:

            pred = model(masked_sample)[0]
            neg_score = pred[0]["score"] 
            pos_score = pred[1]["score"]
            #print(f"neg_score: {neg_score}, pos_score: {pos_score}")
            if neg_score > pos_score:
                y_pred.append(0)
            else:
                y_pred.append(1)


        # Evaluate performance

        perf_val = metric(y_true, y_pred)
        performances.append(perf_val)
    return fractions_removed.tolist(), performances

def moRF_LeRF_LIME(model, X_test, y_test, lime_explanations, metric=accuracy_score, steps=5, MoRF=True):
    
    n_samples = len(lime_explanations)
    # We'll remove fractions from 0 to 1.0 in 'steps' increments:
    fractions_removed = np.linspace(0, 1, steps + 1)
    performances = []

    # Convert y_true to a numpy array for convenience
    y_true = np.array(y_test[:n_samples])

    for frac in fractions_removed:
        # Build a masked version of the entire dataset, one sample at a time
        X_masked_list = []
        for i in range(n_samples):
            # Get the explanation for the i-th sample
            lime_explanation = lime_explanations[i].as_list()
            sorted_lime_explanation = sorted(lime_explanation, key=lambda x: x[1], reverse=MoRF)
            # Get the top features to remove
            n_features_to_remove = int(len(sorted_lime_explanation) * frac)
            features_to_remove = [feature[0] for feature in sorted_lime_explanation[:n_features_to_remove]]
            # Create a masked version of the i-th sample
            X_masked = X_test[i]
            for feature in features_to_remove:
                # Replace the feature with a space
                X_masked = X_masked.replace(feature, " ")
            X_masked_list.append(X_masked)

        y_pred = []
        for masked_sample in X_masked_list:

            pred = model(masked_sample)[0]
            neg_score = pred[0]["score"] 
            pos_score = pred[1]["score"]
            #print(f"neg_score: {neg_score}, pos_score: {pos_score}")
            if neg_score > pos_score:
                y_pred.append(0)
            else:
                y_pred.append(1)
        
        # Evaluate performance

        perf_val = metric(y_true, y_pred)
        performances.append(perf_val)
    return fractions_removed.tolist(), performances