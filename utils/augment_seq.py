import random
import numpy as np

def preprocess_qr(questions, responses, seq_len, pad_val=-1):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )

    return preprocessed_questions, preprocessed_responses