import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import torch
from torch.autograd import Variable
from torch import nn
from transformers import BertModel,BertConfig,AutoModel,RobertaForSequenceClassification


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
        self.init_weight()
        
    def init_weight(self):
        self.weight.data.uniform_(-0.1,0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1,0.1)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class KGNN(nn.Module):
    def __init__(self, bert_model, word_emb_dim = 768, lstm_hid_dim = 300, dropout_rate=0.5, add_dep=False):
        super(KGNN, self).__init__()

        if torch.cuda.is_available():
            self.bert= AutoModel.from_pretrained(bert_model)
        else:
            self.bert = AutoModel.from_pretrained(bert_model)
        self.text_lstm = DynamicLSTM(word_emb_dim, lstm_hid_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*lstm_hid_dim, 2*lstm_hid_dim)
        self.gc2 = GraphConvolution(2*lstm_hid_dim, 2*lstm_hid_dim)
        if add_dep:
            self.fc_out = nn.Linear(2*lstm_hid_dim, 2)
        else:
            self.fc_out = nn.Linear(2*lstm_hid_dim, 2)
        self.text_embed_dropout = nn.Dropout(dropout_rate)
        self.add_dep = add_dep
        self.cls_out = nn.Linear(word_emb_dim, 2)
        self.g = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.g.data.uniform_(0, 1)

    def location_feature(self, feature, offset):
        if torch.cuda.is_available():
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_().cuda()
        else:
            weight = torch.FloatTensor(feature.shape[0], feature.shape[1]).zero_()
        for i in range(offset.shape[0]):
            weight[i] = offset[i][:feature.shape[1]]
        feature = weight.unsqueeze(2) * feature
        return feature

    def forward(self, input_ids, attention_mask, adj=None, offset=None, mask=None):
        # offset是一个位置的加权？
        # mask是aspect的掩码，目前没什么用
        text_len = torch.sum(input_ids != 0, dim=-1).cpu()
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        # print(f"text{text}")
        cls = text[:,0,:].squeeze(1)
        if offset!=None:
            text = self.location_feature(text, offset)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        # text_out: bs*seq_len*lstm_out

        #####    syntactic_level   ######
        if self.add_dep:
            seq_len = text_out.shape[1]
            adj = adj[:, :seq_len, :seq_len]
            x = self.gc1(text_out, adj)
            x = self.gc2(x, adj)
            # alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
            # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
            # x = torch.matmul(alpha, text_out).squeeze(1)
            # x=F.relu(x)
            # print(f"x:{x}")
            x = F.max_pool1d(x.transpose(1,2),x.shape[1]).squeeze(2)
        #####    syntactic_level   ######

        #####    context_level   ######
        # self_socre=torch.bmm(text_out,text_out.transpose(1,2))
        # self_socre=F.softmax(self_socre,dim=1)
        # y=torch.bmm(self_socre,text_out)
        y=text_out
        y=F.max_pool1d(y.transpose(1,2),y.shape[1]).squeeze(2)
        # print(f"y:{y}")
        #####    context_level   ######
        ##### sentence_level #####
        out_c = F.softmax(self.cls_out(cls))
        ##########################

        #####    knowledge_level  ######
        ########...
        ####    knowledge_level  ######

        ######## feature fuse  #########
        if self.add_dep:
            out_xy=torch.cat((x,y),dim=-1)
            # out_xy= self.g*x+(1-self.g)*y
            output=F.softmax(self.fc_out(x))
        else:
            output=F.softmax(self.fc_out(x))
        output = F.softmax(self.g*output+(1-self.g)*out_c)
        return output

class onlyBert(nn.Module):
    def __init__(self, bert_model,word_emb_dim=768):
        super(onlyBert, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(bert_model)
        # self.bert = AutoModel.from_pretrained(bert_model)
        self.fc_out = nn.Linear(word_emb_dim, 2)
    def forward(self, input_ids, attention_mask, adj=None, offset=None, mask=None):
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)#['last_hidden_state'][:,0,:]
        # output = F.softmax(self.fc_out(text))
        output = text.logits
        return output



       