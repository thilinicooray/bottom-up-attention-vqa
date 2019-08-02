import torch
from torch import nn
import torchvision as tv

from modules.embeddings import BiLSTMTextEmbedding
from modules.layers import (BCNet, BiAttention, FCNet,
                                   WeightNormClassifier)
from language_model import WordEmbedding
import utils_imsitu

class resnet_modified_medium(nn.Module):
    def __init__(self):
        super(resnet_modified_medium, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=True)
        #probably want linear, relu, dropout
        self.dropout2d = nn.Dropout2d(.5)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #x = self.dropout2d(x)
        #print('resnet out' , x.size())

        return x


class BAN(nn.Module):
    def __init__(self, dataset, encoder, num_ans_classes):
        super(BAN, self).__init__()
        self._build_word_embedding(dataset)
        self._init_text_embedding()
        self._init_classifier(num_ans_classes)
        self._init_bilinear_attention()
        self.encoder = encoder
        self.convnet = resnet_modified_medium()


    def _build_word_embedding(self, dataset):
        #text_processor = registry.get(self._datasets[0] + "_text_processor")
        #vocab = text_processor.vocab
        #self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)
        self.word_embedding = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)

    def _init_text_embedding(self):
        q_mod = BiLSTMTextEmbedding(
            512,
            300,
            1,
            0.0,
            False,
            'GRU',
        )
        self.q_emb = q_mod

    def _init_bilinear_attention(self):
        num_hidden = 512
        v_dim = 2048

        v_att = BiAttention(v_dim, num_hidden, num_hidden, 4)

        b_net = []
        q_prj = []

        for i in range(4):
            b_net.append(
                BCNet(
                    v_dim, num_hidden, num_hidden, None, k=1
                )
            )

            q_prj.append(
                FCNet(
                    dims=[num_hidden, num_hidden],
                    act="ReLU",
                    dropout=0.2,
                )
            )

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.v_att = v_att

    def _init_classifier(self,num_ans_classes):
        num_hidden = 512
        dropout = 0.5
        self.classifier = WeightNormClassifier(
            num_hidden, num_ans_classes, num_hidden * 2, dropout
        )

    def forward(self, v, labels, gt_verb):

        loss = None

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)
        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        q = self.encoder.get_detailed_roleq_idx(gt_verb, None)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.word_embedding(q)

        q_emb = self.q_emb.forward_all(w_emb)

        b_emb = [0] * 4
        att, logits = self.v_att.forward_all(img, q_emb)
        print('biatt ', att.size(), logits.size())

        for g in range(4):
            g_att = att[:, g, :, :]
            b_emb[g] = self.b_net[g].forward_with_weights(img, q_emb, g_att)
            print('glipz ', q_emb.size(), b_emb[g].size())
            a = self.q_prj[g](b_emb[g].unsqueeze(1))
            print('out q proj', a.size())
            q_emb = a + q_emb

        #print('final size ', q_emb.size())
        logits = self.classifier(q_emb.sum(1))

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels):

        batch_size = role_label_pred.size()[0]

        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                for j in range(0, self.encoder.max_role_count):
                    frame_loss += utils_imsitu.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.encoder.get_num_labels())
                frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                #print('frame loss', frame_loss, 'verb loss', verb_loss)
                loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss