import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import torch
from torch import nn
from transformers import BertModel,BertConfig


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class KGNN(nn.Module):
    def __init__(self, bert_model, word_emb_dim = 768, lstm_hid_dim = 300, dropout_rate=0.5):
        super(KGNN, self).__init__()

        if torch.cuda.is_available():
            self.bert= BertModel.from_pretrained(bert_model)
        else:
            self.bert = BertModel.from_pretrained(bert_model)
        self.text_lstm = DynamicLSTM(word_emb_dim, lstm_hid_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*lstm_hid_dim, 2*lstm_hid_dim)
        self.gc2 = GraphConvolution(2*lstm_hid_dim, 2*lstm_hid_dim)
        self.fc_out = nn.Linear(4*lstm_hid_dim, 2)
        self.text_embed_dropout = nn.Dropout(dropout_rate)

    def location_feature(self, feature, offset):
        if torch.cuda.is_available():
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_().cuda()
        else:
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i] = offset[i][:feature.shape[1]]
        feature = weight.unsqueeze(2) * feature
        return feature

    def forward(self, input_ids, attention_mask, adj, offset=None, mask=None):
        # offset是一个位置的加权？
        # mask是aspect的掩码，目前没什么用
        text_len = torch.sum(input_ids != 0, dim=-1).cpu()
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        if offset!=None:
            text = self.location_feature(text, offset)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)

        #####    syntactic_level   ######
        seq_len = text_out.shape[1]
        adj = adj[:, :seq_len, :seq_len]
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)
        x=F.relu(x)
        #####    syntactic_level   ######

        #####    context_level   ######
        self_socre=torch.bmm(text_out,text_out.transpose(1,2))
        self_socre=F.softmax(self_socre,dim=1)
        y=torch.bmm(self_socre,text_out)
        y=F.relu(F.max_pool1d(y.transpose(1,2),y.shape[1]).squeeze(2))
        #? 直接把序列维度pool掉了
        #####    context_level   ######

        #####    knowledge_level  ######
        ########...
        ####    knowledge_level  ######

        ######## feature fuse  #########
        out_xy=torch.cat((x,y),dim=-1)
        output=F.softmax(self.fc_out(out_xy))
        return output