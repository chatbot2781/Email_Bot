
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MLService:
    """
    Loads ML artifacts once and provides generate_response() decision:
      - If ML not available or similarity < threshold → CREATE_SERVICENOW_INCIDENT
      - Else → AUTO_RESPONSE with stored response text
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.faiss_index = None
        self.df = None
        self.model = None
        self._load()

    def _load(self):
        try:
            index_path = os.path.join(self.model_dir, "email_support_index.faiss")
            df_path = os.path.join(self.model_dir, "email_support_data.pkl")
            model_path = os.path.join(self.model_dir, "embedding_model")

            self.faiss_index = faiss.read_index(index_path)
            with open(df_path, "rb") as f:
                self.df = pickle.load(f)
            self.model = SentenceTransformer(model_path)
        except Exception:
            # If anything fails, keep ML unavailable (fallback to incident creation)
            self.faiss_index = None
            self.df = None
            self.model = None

    def available(self) -> bool:
        return self.model is not None and self.faiss_index is not None and self.df is not None

    def generate_response(self, email_body: str, threshold: float = 0.75):
        if not email_body or not email_body.strip():
            return {"action": "CREATE_SERVICENOW_INCIDENT", "similarity": 0.0}

        if not self.available():
            return {"action": "CREATE_SERVICENOW_INCIDENT", "similarity": 0.0}

        embedding = self.model.encode([email_body])
        D, I = self.faiss_index.search(embedding, 1)
        # Use a simple transform of distance to a [0,1] similarity score
        similarity = float(1 / (1 + float(D[0][0])))

        if similarity < threshold:
            return {"action": "CREATE_SERVICENOW_INCIDENT", "similarity": similarity}

        response = str(self.df.iloc[int(I[0][0])]["email_response"])
        return {
            "action": "AUTO_RESPONSE",
            "similarity": similarity,
            "response": response,
        }
