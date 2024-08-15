import torch
from transformers import BertModel


class DTW(torch.nn.Module):
    """
    Model that uses DTW to check similarity between two sequence of sentence.
    """
    def __init__(self,pre_trained_model_name, crossing = True):
        super(DTW, self).__init__()
        self.crossing = crossing
        self.bert_model = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)

        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.cos = torch.nn.CosineSimilarity(dim = -1)
        self.a = torch.nn.Parameter(torch.rand(1, requires_grad = True, dtype = torch.float))
        self.b = torch.nn.Parameter(torch.rand(1, requires_grad = True, dtype = torch.float))
        self.tanh = torch.nn.Tanh()

    def forward(self, input_ids_1 = None, attention_mask_1 = None,
                        token_type_ids_1 = None, input_ids_2 = None,
                        attention_mask_2 = None, token_type_ids_2 = None, test = False):
        """
        Forward of this model that takes sentence as separate input and predict the similarity score using DTW.
        """
        output1 = self.bert_model(input_ids = input_ids_1, attention_mask = attention_mask_1, token_type_ids = token_type_ids_1)[0]
        output2 = self.bert_model(input_ids = input_ids_2, attention_mask = attention_mask_2, token_type_ids = token_type_ids_2)[0]
        sim_scores = []
        for i in range(len(output1)):
            sim_score = self.get_DTW_score(output1[i][attention_mask_1[i] == 1][1:-1], output2[i][attention_mask_2[i] == 1][1:-1],
                                           return_map = False, crossing = self.crossing)
            sim_scores.append(sim_score)   
        
        return torch.cat(sim_scores)

    def score(self,s1_i, s2_j, eps=1e-8):
        """
        Given two list of word embeddings find the cosine similarity between every pair of tokens in both sentence.
        """
        a_n, b_n = s1_i.norm(dim=1)[:, None], s2_j.norm(dim=1)[:, None]
        a_norm = s1_i / torch.clamp(a_n, min=eps)
        b_norm = s2_j / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return self.tanh(sim_mt * self.a + self.b)
    
    def get_DTW_score(self, s1, s2, return_map = False, crossing = False):
        """
        Method that helps in getting DTW similarity score given embeddings of tokens in sentences.
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        I = len(s1)
        J = len(s2)
        sim_mat = self.score(s1, s2)

        if crossing:
            sim = torch.sum(torch.max(sim_mat, dim = 0)[0]) / J
            if return_map:
                k = torch.argmax(sim_mat,dim=0)
                return sim.reshape(1), k
            return sim.reshape(1)
        else:
            M = sim_mat > 0
            P = torch.zeros((I,J))
            K = torch.zeros((I,J), dtype = torch.int)
            for i in range(I):
                P[i][0] = sim_mat[i][0]
            K[:,0] = -1

            for j in range(1, J):
                max_val = float('-inf')
                ptr = None
                P[0][j] = max(0, sim_mat[0][j])
                for i in range(1,I):
                    if max_val < P[i-1][j-1]:
                        max_val = P[i-1][j-1]
                        ptr = i-1
                    P[i][j] = max_val + max(0, sim_mat[i][j])
                    K[i][j] = ptr

            # print("sim_mat")
            # print(sim_mat)
            # print("P")
            # print(P)
            # print("M")
            # print(M)
            # print("K")
            # print(K)

            if return_map:
                m = [None] * J
                k = [None] * J
                I_prime = int(torch.argmax(P,dim=0)[J-1])
                m[-1] = I_prime
                K[I_prime][0] = I_prime

                if M[I_prime][J-1]:
                    k[-1] = m[-1]
                else:
                    k[-1] = None

                for j in range(J-2, -1, -1):
                    m[j] = K[m[j+1]][j+1]
                    if m[j] == -1 or m[j] == None:
                        break
                    if M[m[j]][j]:
                        k[j] = m[j].item()
                    else:
                        k[j] = None
                return P[I-1][J-1].reshape(1), k
            
            return P[I-1][J-1].reshape(1) / J
