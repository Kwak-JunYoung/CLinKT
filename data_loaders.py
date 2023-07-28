from torch.utils.data import Dataset
from utils.augment_seq import preprocess_qr
import torch
from collections import defaultdict
import numpy as np


def get_diff_df(df, seq_len, num_skills, num_questions, total_cnt_init=1, diff_unk=0.0):
    q_total_cnt = np.ones((num_questions+1))
    q_crt_cnt = np.zeros((num_questions+1))

    c_total_cnt = np.ones((num_skills+1))
    c_crt_cnt = np.zeros((num_skills+1))
    
    if total_cnt_init == 0:
        q_total_cnt = np.zeros((num_questions+1))
        c_total_cnt = np.zeros((num_skills+1))

    for q, c, r in zip(df["item_id"], df["skill_id"], df["correct"]):
        c_total_cnt[c] += 1
        if r:
            c_crt_cnt[c] += 1
        q_total_cnt[q] += 1
        if r:
            q_crt_cnt[q] += 1

    if diff_unk != 0.0: ## else unk is zero
        q_crt_cnt[np.where(q_total_cnt == total_cnt_init)] = diff_unk
        c_crt_cnt[np.where(c_total_cnt == total_cnt_init)] = diff_unk

    if total_cnt_init == 0:
        q_total_cnt = np.where(q_total_cnt == 0, 1, q_total_cnt)
        c_total_cnt = np.where(c_total_cnt == 0, 1, c_total_cnt)

    q_diff = q_crt_cnt/q_total_cnt
    c_diff = c_crt_cnt/c_total_cnt
    df = df.assign(item_diff=q_diff[np.array(df["item_id"].values)])
    df = df.assign(skill_diff=c_diff[np.array(df["skill_id"].values)])
    
    print("-"*80)
    s_df = df.loc[:, ['skill_id', 'skill_diff']]
    s_df = s_df.drop_duplicates(subset=["skill_id"]).sort_values(by='skill_id')
    q_df = df.loc[:, ['item_id', 'item_diff']]
    q_df = q_df.drop_duplicates(subset=["item_id"]).sort_values(by='item_id')

    responses = [
        u_df["correct"].values[-seq_len :]
        for _, u_df in df.groupby("user_id")
    ]
    
    print(f"mean of total set, skill correct ratio:{np.mean(s_df['skill_diff']*100):.2f}")
    print(f"mean of total set, question correct ratio:{np.mean(q_df['item_diff']*100):.2f}")
    print(f"mean of total set, class 0 ratio:{sum(list(x).count(0) for x in responses)/sum(len(x) for x in responses):.2f}")
    print(f"mean of total set, class 1 ratio:{sum(list(x).count(1) for x in responses)/sum(len(x) for x in responses):.2f}")
    print("-"*80)

    return df

class MostRecentQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[-self.seq_len:]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[-self.seq_len:]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[-self.seq_len:]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        question_correct = defaultdict(int)
        question_count = defaultdict(int)
        
        for q_list, s_list, r_list in zip(self.questions, self.skills, self.responses):
            for q, s, r in zip(q_list, s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1
                question_correct[q] += r
                question_count[q] += 1

        skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }
        self.ordered_skills = [
            item[0] for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]
        question_difficulty = {
            q: question_correct[q] / float(question_count[q]) for q in question_correct
        }

        self.sdiff_array = np.zeros(self.num_skills+1)
        self.qdiff_array = np.zeros(self.num_questions+1)
        self.sdiff_array[list(skill_difficulty.keys())] = np.array(
            list(skill_difficulty.values()))
        self.qdiff_array[list(question_difficulty.keys())] = np.array(
            list(question_difficulty.values()))

        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(self.ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = self.ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(self.ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = self.ordered_skills[i - 1]
            else:
                self.easier_skills[s] = self.ordered_skills[i + 1]
                self.harder_skills[s] = self.ordered_skills[i - 1]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )
        self.padded_sd = torch.full(
            (len(self.skills), self.seq_len), -1, dtype=torch.float
        )
        self.padded_qd = torch.full(
            (len(self.questions), self.seq_len), -1, dtype=torch.float
        )
        self.position = torch.full(
            (len(self.questions), self.seq_len), 0, dtype=torch.long
        )

    def __getitem__(self, index):

        q, s, r = self.questions[index], self.skills[index], self.responses[index]
        sd = self.sdiff_array[s]
        qd = self.qdiff_array[q]
        self.padded_q[index, -len(q):] = torch.tensor(q, dtype=torch.long)
        self.padded_s[index, -len(s):] = torch.tensor(s, dtype=torch.long)
        self.padded_r[index, -len(r):] = torch.tensor(r, dtype=torch.long)
        self.attention_mask[index, -
                            len(s):] = torch.ones(len(s), dtype=torch.long)
        self.padded_sd[index, -len(s):] = torch.tensor(sd, dtype=torch.float)
        self.padded_qd[index, -len(q):] = torch.tensor(qd, dtype=torch.float)
        self.position[index, -len(s):] = torch.arange(1,
                                                      len(s)+1, dtype=torch.long)

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "attention_mask": self.attention_mask[index],
            "sdiff": self.padded_sd[index],
            "qdiff": self.padded_qd[index],
            "position": self.position[index],
        }

    def __len__(self):
        return self.len


class MostEarlyQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses)):
            q, s, r = elem
            self.padded_q[i, : len(q)] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            self.attention_mask[i, : len(r)] = torch.ones(
                len(s), dtype=torch.long)

    def __getitem__(self, index):
        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class SkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["skill_id"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.questions, self.responses = preprocess_qr(
            self.questions, self.responses, self.seq_len
        )
        self.len = len(self.questions)

    def __getitem__(self, index):
        return {"questions": self.questions[index], "responses": self.responses[index]}

    def __len__(self):
        return self.len
