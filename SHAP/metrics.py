import numpy as np
from sklearn.metrics import accuracy_score



def moRF_LeRF_variable_length(
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