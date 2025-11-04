import numpy as np
import matplotlib.pyplot as plt
from src.config import VERIFICATION_THRESHOLD
from src.data_preparation import load_and_preprocess_image

def evaluate_random_pair(feature_extractor, reference_dict, distorted_dict, person_ids):
    """Verifies random person and displays comparison."""
    import random, os
    TEST_ID = random.choice(person_ids)
    ref_path = reference_dict[TEST_ID][0]
    query_path = distorted_dict[TEST_ID][0]
    ref_img = load_and_preprocess_image(ref_path)
    query_img = load_and_preprocess_image(query_path)
    emb_ref = feature_extractor.predict(np.expand_dims(ref_img, 0))[0]
    emb_query = feature_extractor.predict(np.expand_dims(query_img, 0))[0]
    dist = np.linalg.norm(emb_ref - emb_query)
    verdict = "Same Person ✅" if dist < VERIFICATION_THRESHOLD else "Different Person ❌"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Verification Result: {verdict}", color='green' if dist < VERIFICATION_THRESHOLD else 'red')
    axes[0].imshow(ref_img)
    axes[0].set_title(f"Reference: {os.path.basename(ref_path)}")
    axes[1].imshow(query_img)
    axes[1].set_title(f"Query: {os.path.basename(query_path)}\nDistance: {dist:.4f}")
    plt.show()
