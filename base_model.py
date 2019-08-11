import torch
import torch.nn as nn
from attention import Attention, NewAttention, BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, FCNet_relu, MLP, BCNet
import torchvision as tv
import utils_imsitu
import numpy as np
import torch.nn.functional as F
import math

import copy

class resnet_152_features(nn.Module):
    def __init__(self):
        super(resnet_152_features, self).__init__()
        self.resnet = tv.models.resnet152(pretrained=True)

        #probably want linear, relu, dropout

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

        return x

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


class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = vgg.features

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        features = self.vgg_features(x)

        return features


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class BaseModelGrid(nn.Module):
    def __init__(self, conv_net, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModelGrid, self).__init__()
        self.conv_net = conv_net
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        img_features = self.conv_net(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img = img_features.view(batch_size, n_channel, -1)
        v = img.permute(0, 2, 1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class BaseModelGrid_Imsitu(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder):
        super(BaseModelGrid_Imsitu, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder

    def forward(self, v, q, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

        q = q.view(v.size(0)* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return role_label_pred

    def forward_noq(self, v, verb):
        q = self.encoder.get_role_q_by_verb(verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_pred = self.forward(v, q, None)

        return role_label_pred

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels,args):

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

class BaseModelGrid_Imsitu_Agent(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_ans_classes):
        super(BaseModelGrid_Imsitu_Agent, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_ans_classes = num_ans_classes

    def forward(self, v, q, labels, verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        img = v
        q = q.squeeze()
        #print('inside model :', img.size(), q.size())

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        loss = None
        if self.training:
            loss = self.calculate_loss(logits, labels)

        return logits, loss

    def forward_noq(self, v, verb):
        q = self.encoder.get_role_q_by_verb(verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_pred = self.forward(v, q, None)

        return role_label_pred

    def calculate_loss(self, role_label_pred, gt_labels):

        batch_size = role_label_pred.size()[0]

        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                loss += utils_imsitu.cross_entropy_loss(role_label_pred[i], gt_labels[i,index] ,self.num_ans_classes)

        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_RoleIter(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter):
        super(BaseModelGrid_Imsitu_RoleIter, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter

    def forward(self, v, q, labels, gt_verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        losses = []
        prev_rep = None
        batch_size = v.size(0)
        for i in range(self.num_iter):

            img = v

            img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
            q = q.view(batch_size* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = joint_repr + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            if self.training:
                losses.append(self.calculate_loss(gt_verb, role_label_pred, labels))

            label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return role_label_pred, loss

    def forward_eval(self, v, q, labels, gt_verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        losses = []
        prev_rep = None
        for i in range(self.num_iter):

            img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

            q = q.view(v.size(0)* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = joint_repr + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)


            label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()

            verb_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = verb_q_idx.to(torch.device('cuda'))

        loss = None


        return role_label_pred, loss

    '''def forward(self, v, labels, gt_verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()

        q = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        q = q.view(v.size(0)* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        loss = None

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss'''

    def forward_noq(self, v, verb):
        q = self.encoder.get_role_q_by_verb(verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_pred, _ = self.forward_eval(v, q, None, verb)

        return role_label_pred

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


class BaseModelGrid_Imsitu_RoleIter_With_CNN(nn.Module):
    def __init__(self, convnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter):
        super(BaseModelGrid_Imsitu_RoleIter_With_CNN, self).__init__()
        self.convnet = convnet
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter
        self.dropout = nn.Dropout(0.3)

    def forward_gt(self, v, q1, labels, gt_verb):

        loss = None

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()

        role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss


    def forward(self, v, q, labels, gt_verb):

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        for i in range(self.num_iter):

            img = v

            img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
            q = q.view(batch_size* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            if self.training:
                losses.append(self.calculate_loss(gt_verb, role_label_pred, labels))

            label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return role_label_pred, loss

    def forward_all_roles_with_rep(self, v, q):

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr


        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)
        role_rep = joint_repr.contiguous().view(v.size(0), self.encoder.max_role_count, -1)


        return role_rep, role_label_pred

    def forward_agent_place_only(self, v, q, gt_verb=None, is_general=False):
        role_count = 2

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        #todo only 1 iter possible for now. handle it
        for i in range(self.num_iter):

            img = v

            img = img.expand(role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * role_count, -1, v.size(2))
            q = q.view(batch_size* role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), role_count, -1)

            label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()


        return role_label_pred

    def forward_agent_place_only_with_rep(self, v, q, gt_verb=None, is_general=False):
        role_count = 2

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        #todo only 1 iter possible for now. handle it
        for i in range(self.num_iter):

            img = v

            img = img.expand(role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * role_count, -1, v.size(2))
            q = q.view(batch_size* role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), role_count, -1)
            joint_repr_out = joint_repr.contiguous().view(v.size(0), role_count, -1)

            #label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()


        return joint_repr_out, role_label_pred

    def forward_noq(self, v, verb=None):
        q = self.encoder.get_agent_place_roleqs(v.size(0), verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        if verb is None:
            role_label_pred = self.forward_agent_place_only(v, q, None, False)
        else:
            role_label_pred = self.forward_agent_place_only(v, q, verb, True)

        return role_label_pred

    def forward_noq_reponly(self, v, verb=None):
        q = self.encoder.get_agent_place_roleqs(v.size(0), verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        if verb is None:
            role_rep, role_label_pred = self.forward_agent_place_only_with_rep(v, q, None, False)
        else:
            role_rep, role_label_pred = self.forward_agent_place_only_with_rep(v, q, verb, True)

        return role_rep, role_label_pred

    def forward_noq_reponly_allroles(self, v, verb=None):
        q = self.encoder.get_detailed_roleq_idx(verb, None)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_rep, role_label_pred = self.forward_all_roles_with_rep(v, q)

        return role_rep, role_label_pred

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

class BaseModelGrid_Imsitu_RoleIter_With_CNN_EXTCTX(nn.Module):
    def __init__(self, convnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter):
        super(BaseModelGrid_Imsitu_RoleIter_With_CNN_EXTCTX, self).__init__()
        self.convnet = convnet
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter
        self.dropout = nn.Dropout(0.1)
        self.resize_ctx = nn.Linear(1024, 2048)
        self.decoder = MLP(1024,1024,2048, num_layers=1, dropout_p=0.1)
        self.l2_criterion = nn.MSELoss()
        self.Dropout_M = nn.Dropout(0.1)

        self.ctx_att = MultiHeadedAttention(4, 1024, dropout=0.1)

        '''self.longq_embd = FCNet([1024*4, 1024])
        self.g = nn.Sequential(
            nn.Linear(1024*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )'''

    def forward_gt(self, v, labels, gt_verb):

        loss = None

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()

        role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss


    def forward_pred(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        img_feat_flat = img_feat_flat.expand(self.encoder.max_role_count,img_feat_flat.size(0), img_feat_flat.size(1))
        img_feat_flat = img_feat_flat.transpose(0,1)
        img_feat_flat = img_feat_flat.contiguous().view(-1, img_feat_flat.size(-1))
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_rep, role_pred = self.ctx_role_model.forward_noq_reponly_allroles(v_org, gt_verb)

        role_rep_expand = role_rep.expand(self.encoder.max_role_count, role_rep.size(0), role_rep.size(1), role_rep.size(2))
        role_rep_expand = role_rep_expand.transpose(0,1)
        role_rep_expand_new = torch.zeros([batch_size, self.encoder.max_role_count, self.encoder.max_role_count-1, role_rep.size(2)])
        for i in range(self.encoder.max_role_count):
            if i == 0:
                role_rep_expand_new[:,i] = role_rep_expand[:,i,1:]
            elif i == self.encoder.max_role_count -1:
                role_rep_expand_new[:,i] = role_rep_expand[:,i,:i]
            else:
                role_rep_expand_new[:,i] = torch.cat([role_rep_expand[:,i,:i], role_rep_expand[:,i,i+1:]], 1)

        if torch.cuda.is_available():
            role_rep_expand_new = role_rep_expand_new.to(torch.device('cuda'))


        role_rep_combo = torch.sum(role_rep_expand_new, 2)
        role_rep_combo = role_rep_combo.view(-1, role_rep_combo.size(-1))
        ext_ctx = img_feat_flat * role_rep_combo


        label_idx = torch.max(role_pred,-1)[1]
        role_q_idx = self.encoder.get_detailed_roleq_idx_agentplace_ctx(gt_verb, label_idx)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = v_repr + ext_ctx

        rep_img = joint_repr * q_repr

        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        loss = None

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels) + self.l2_criterion(rep_img, img_feat_flat)


        return role_label_pred, loss

    def forward_extctx(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        img_feat_flat = img_feat_flat.expand(self.encoder.max_role_count,img_feat_flat.size(0), img_feat_flat.size(1))
        img_feat_flat = img_feat_flat.transpose(0,1)
        img_feat_flat = img_feat_flat.contiguous().view(-1, img_feat_flat.size(-1))
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_rep, role_pred = self.ctx_role_model.forward_noq_reponly_allroles(v_org, gt_verb)

        role_rep_expand = role_rep.expand(self.encoder.max_role_count, role_rep.size(0), role_rep.size(1), role_rep.size(2))
        role_rep_expand = role_rep_expand.transpose(0,1)
        role_rep_expand_new = torch.zeros([batch_size, self.encoder.max_role_count, self.encoder.max_role_count-1, role_rep.size(2)])
        for i in range(self.encoder.max_role_count):
            if i == 0:
                role_rep_expand_new[:,i] = role_rep_expand[:,i,1:]
            elif i == self.encoder.max_role_count -1:
                role_rep_expand_new[:,i] = role_rep_expand[:,i,:i]
            else:
                role_rep_expand_new[:,i] = torch.cat([role_rep_expand[:,i,:i], role_rep_expand[:,i,i+1:]], 1)

        if torch.cuda.is_available():
            role_rep_expand_new = role_rep_expand_new.to(torch.device('cuda'))


        role_rep_combo = torch.sum(role_rep_expand_new, 2)
        role_rep_combo = role_rep_combo.view(-1, role_rep_combo.size(-1))
        ext_ctx = img_feat_flat * role_rep_combo


        label_idx = torch.max(role_pred,-1)[1]
        role_q_idx = self.encoder.get_detailed_roleq_idx_agentplace_ctx(gt_verb, label_idx)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr + ext_ctx


        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        loss = None

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)


        return role_label_pred, loss

    def forward_noctx(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)

        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_q_idx = self.encoder.get_role_nl_questions_batch(gt_verb)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr


        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        loss = None

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)


        return role_label_pred, loss

    def forward_relnet(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)

        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        img_feat_flat = img_feat_flat.expand(self.encoder.max_role_count,img_feat_flat.size(0), img_feat_flat.size(1))
        img_feat_flat = img_feat_flat.transpose(0,1)
        img_feat_flat = img_feat_flat.contiguous().view(-1, img_feat_flat.size(-1))

        batch_size = v.size(0)

        role_q_idx = self.encoder.get_role_nl_questions_batch(gt_verb)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)
        #labels = labels.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        n_heads = 4
        img_mul_head = img.view(img.size(0), img.size(1),  n_heads, -1).transpose(1, 2)
        img_mul_head = img_mul_head.contiguous().view(-1, img_mul_head.size(2), img_mul_head.size(-1))

        q_emb_mul_head = q_emb.view(q_emb.size(0), n_heads, -1)
        q_emb_mul_head = q_emb_mul_head.contiguous().view(-1, q_emb_mul_head.size(-1))

        #print('img q :', img_mul_head.size(), q_emb_mul_head.size())

        att = self.v_att(img_mul_head, q_emb_mul_head)
        v_emb = (att * img_mul_head).sum(1) # [batch, v_dim]
        #v_emb = v_emb.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        q_repr = self.q_net(q_emb_mul_head)
        v_repr = self.v_net(v_emb)

        qst_org = self.longq_embd(q_repr.contiguous().view(batch_size* self.encoder.max_role_count, -1))
        qst = torch.unsqueeze(qst_org, 1)
        qst = qst.repeat(1,n_heads * n_heads,1)
        qst = torch.squeeze(qst)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)

        #mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
        #mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        #mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        #mfb_sign_sqrt = torch.sqrt(F.relu(mfb_iq_drop)) - torch.sqrt(F.relu(-mfb_iq_drop))
        #mfb_l2 = F.normalize(mfb_sign_sqrt)

        subans_grouped = mfb_iq_drop.contiguous().view(batch_size * self.encoder.max_role_count, n_heads, -1)
        sub_ans1 = subans_grouped.unsqueeze(1).expand(batch_size * self.encoder.max_role_count, n_heads, n_heads, mfb_iq_drop.size(-1))
        sub_ans2 = subans_grouped.unsqueeze(2).expand(batch_size * self.encoder.max_role_count, n_heads, n_heads, mfb_iq_drop.size(-1))

        sub_ans1 = sub_ans1.contiguous().view(-1, n_heads * n_heads, mfb_iq_drop.size(-1))
        sub_ans2 = sub_ans2.contiguous().view(-1, n_heads * n_heads, mfb_iq_drop.size(-1))
        #print('size :', sub_ans2.size(), qst.size())
        concat_vec = torch.cat([sub_ans1, sub_ans2, qst], 2).view(-1, sub_ans1.size(-1)*3)

        lin1out = self.g(concat_vec)

        '''mfb_iq_resh = val.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads * n_heads)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)'''
        compositionedans = lin1out.view(-1, n_heads * n_heads, sub_ans1.size(-1)).sum(1).squeeze()

        logits = self.classifier(qst_org * compositionedans)

        loss = None
        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)
        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss

    def forward_agent_place_only(self, v, q, gt_verb=None, is_general=False):
        role_count = 2

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        #todo only 1 iter possible for now. handle it
        for i in range(self.num_iter):

            img = v

            img = img.expand(role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * role_count, -1, v.size(2))
            q = q.view(batch_size* role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), role_count, -1)

            label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()


        return role_label_pred

    def forward_agent_place_only_with_rep(self, v, q, gt_verb=None, is_general=False):
        role_count = 2

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        #todo only 1 iter possible for now. handle it
        for i in range(self.num_iter):

            img = v

            img = img.expand(role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * role_count, -1, v.size(2))
            q = q.view(batch_size* role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), role_count, -1)
            joint_repr_out = joint_repr.contiguous().view(v.size(0), role_count, -1)

            #label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()


        return joint_repr_out, role_label_pred

    def forward_noq(self, v, verb=None):
        q = self.encoder.get_agent_place_roleqs(v.size(0), verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        if verb is None:
            role_label_pred = self.forward_agent_place_only(v, q, None, False)
        else:
            role_label_pred = self.forward_agent_place_only(v, q, verb, True)

        return role_label_pred

    def forward_noq_reponly(self, v, verb=None):
        q = self.encoder.get_agent_place_roleqs(v.size(0), verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        if verb is None:
            role_rep, role_label_pred = self.forward_agent_place_only_with_rep(v, q, None, False)
        else:
            role_rep, role_label_pred = self.forward_agent_place_only_with_rep(v, q, verb, True)

        return role_rep, role_label_pred

    def forward_mulheadatt_factpool(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)

        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_q_idx = self.encoder.get_role_nl_questions_batch(gt_verb)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        img_org_all = v
        q = q.view(batch_size* self.encoder.max_role_count, -1)
        #labels = labels.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        for i in range(1):

            n_heads = 4
            img_mul_head = img.view(img.size(0), img.size(1),  n_heads, -1).transpose(1, 2)
            img_mul_head = img_mul_head.contiguous().view(-1, img_mul_head.size(2), img_mul_head.size(-1))

            q_emb_mul_head = q_emb.view(q_emb.size(0), n_heads, -1)
            q_emb_mul_head = q_emb_mul_head.contiguous().view(-1, q_emb_mul_head.size(-1))

            #print('img q :', img_mul_head.size(), q_emb_mul_head.size())
            #attention

            att = self.v_att(img_mul_head, q_emb_mul_head)
            v_emb = (att * img_mul_head).sum(1) # [batch, v_dim]
            #v_emb = v_emb.contiguous().view(batch_size* self.encoder.max_role_count, -1)
            q_repr = self.q_net(q_emb_mul_head)
            v_repr = self.v_net(v_emb)

            #composition

            mfb_iq_eltwise = torch.mul(q_repr, v_repr)

            mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)

            mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
            mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
            mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
            mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
            mfb_l2 = F.normalize(mfb_sign_sqrt)

            #contextualization


            out = mfb_l2


        logits = self.classifier(out)

        loss = None
        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)
        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss

    def forward(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)
        img_feat_flat = self.convnet.resnet.avgpool(img_features).squeeze()
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_q_idx = self.encoder.get_role_nl_questions_batch(gt_verb)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))


        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        img_exp_org = img
        q = q.view(batch_size* self.encoder.max_role_count, -1)
        #labels = labels.view(batch_size* self.encoder.max_role_count, -1)
        n_heads = 4

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        q_emb_mul_head = q_emb.view(q_emb.size(0), n_heads, -1)
        q_emb_mul_head = q_emb_mul_head.contiguous().view(-1, q_emb_mul_head.size(-1))
        q_repr = self.q_net(q_emb_mul_head)
        prev = None

        '''init_vemb = torch.zeros(batch_size * self.encoder.max_role_count * n_heads, v.size(2)//n_heads)
        if torch.cuda.is_available():
            init_vemb = init_vemb.to(torch.device('cuda'))

        vemb_list = [init_vemb]'''

        for i in range(2):


            img_mul_head = img.view(img.size(0), img.size(1),  n_heads, -1).transpose(1, 2)
            img_mul_head = img_mul_head.contiguous().view(-1, img_mul_head.size(2), img_mul_head.size(-1))



            #print('img q :', img_mul_head.size(), q_emb_mul_head.size())
            #attention

            att = self.v_att(img_mul_head, q_emb_mul_head)
            v_emb = (att * img_mul_head).sum(1) # [batch, v_dim]
            #vemb_list.append(v_emb)
            #v_emb = v_emb.contiguous().view(batch_size* self.encoder.max_role_count, -1)
            v_repr = self.v_net(v_emb)

            #composition

            mfb_iq_eltwise = torch.mul(q_repr, v_repr)

            mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)

            mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
            mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
            mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
            mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
            mfb_l2 = F.normalize(mfb_sign_sqrt)

            #contextualization

            cur_group = mfb_l2.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            #print('before att :', cur_group[1,:, :5])
            mask = self.encoder.get_adj_matrix_noself(gt_verb)

            if torch.cuda.is_available():
                mask = mask.to(torch.device('cuda'))

            selfatt_val= self.ctx_att(cur_group, cur_group, cur_group, mask=mask)

            #print('after att :', selfatt_val[1,:, :5])

            withctx = selfatt_val.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)

            img = img * self.resize_ctx(withctx).unsqueeze(1)

            out = mfb_l2
            '''if prev is not None:
                out = prev + self.dropout(out)

            prev = out'''

        ans_with_ctx = torch.sum((out).contiguous().view(v.size(0), self.encoder.max_role_count, -1),1)
        decoded_img = self.decoder(ans_with_ctx)
        logits = self.classifier(out)

        loss = None
        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)
        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels) + self.l2_criterion(decoded_img, img_feat_flat)

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

    def calculate_loss1(self, role_label_pred, gt_labels):
        #loss = nn.KLDivLoss(reduction='sum')
        loss = nn.BCEWithLogitsLoss()
        final_loss = loss(role_label_pred, gt_labels) * role_label_pred.size(1)
        return final_loss

    def calculate_loss_new(self, batch_size, role_label_pred, gt_labels):
        #loss = nn.KLDivLoss(reduction='sum')
        loss = nn.BCEWithLogitsLoss(reduction='sum')
        role_leb_placewise = role_label_pred.view(batch_size, self.encoder.max_role_count, -1)
        gt_labels_placewise = gt_labels.view(batch_size, self.encoder.max_role_count, -1)
        role_loss = 0
        for j in range(0, self.encoder.max_role_count):
            n = 0
            curr_loss = 0
            for i in range(batch_size):
                if not (gt_labels_placewise[i][j][-1] > 0.0) :
                    n += 1
                    curr_loss += loss(role_leb_placewise[i][j], gt_labels_placewise[i][j])
            output = curr_loss / (n + 10e-8)
            role_loss += output
        return (role_loss / self.encoder.max_role_count)

class BaseModelGrid_Imsitu_RoleIter_With_CNN_NewModel(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, classifier, encoder, num_iter):
        super(BaseModelGrid_Imsitu_RoleIter_With_CNN_NewModel, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter
        self.resize_ctx = nn.Linear(3072, 2048)
        self.l2_criterion = nn.MSELoss()
        self.Dropout_M = nn.Dropout(0.1)
        self.Dropout_Q = nn.Dropout(0.1)
        self.Dropout_C = nn.Dropout(0.1)
        #self.context_adder = nn.GRUCell(1024, 1024)
        #self.context_adder = nn.Linear(2048,1024)

        self.ctx_att = MultiHeadedAttention(4, 1024, dropout=0.1)

    def forward_gt(self, v, labels, gt_verb):

        loss = None

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()

        role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        img = v

        img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
        q = q.view(batch_size* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss


    def forward_prev(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)
        #img_feat_flat = self.convnet.resnet.avgpool(img_features).squeeze()
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))


        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        n_heads = 4

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.Dropout_Q(self.query_composer(role_verb_embd))

        q_emb_mul_head = q_emb.view(q_emb.size(0), n_heads, -1)
        q_emb_mul_head = q_emb_mul_head.contiguous().view(-1, q_emb_mul_head.size(-1))
        q_repr = self.q_net(q_emb_mul_head)
        prev = None

        '''init_vemb = torch.zeros(batch_size * self.encoder.max_role_count * n_heads, v.size(2)//n_heads)
        if torch.cuda.is_available():
            init_vemb = init_vemb.to(torch.device('cuda'))

        vemb_list = [init_vemb]'''

        for i in range(2):


            img_mul_head = img.view(img.size(0), img.size(1),  n_heads, -1).transpose(1, 2)
            img_mul_head = img_mul_head.contiguous().view(-1, img_mul_head.size(2), img_mul_head.size(-1))

            #print('img q :', img_mul_head.size(), q_emb_mul_head.size())
            #attention

            att = self.v_att(img_mul_head, q_emb_mul_head)
            v_emb = (att * img_mul_head).sum(1) # [batch, v_dim]
            #vemb_list.append(v_emb)
            #v_emb = v_emb.contiguous().view(batch_size* self.encoder.max_role_count, -1)
            v_repr = self.v_net(v_emb)

            #composition

            mfb_iq_eltwise = torch.mul(q_repr, v_repr)

            mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)

            mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
            mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
            mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
            mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
            mfb_l2 = F.normalize(mfb_sign_sqrt)

            #contextualization

            cur_group = mfb_l2.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            #print('before att :', cur_group[1,:, :5])
            '''mask = self.encoder.get_adj_matrix_noself(gt_verb)

            if torch.cuda.is_available():
                mask = mask.to(torch.device('cuda'))

            selfatt_val= self.ctx_att(cur_group, cur_group, cur_group, mask=mask)

            #print('after att :', selfatt_val[1,:, :5])

            withctx = selfatt_val.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)'''

            cur_group1 = cur_group.unsqueeze(1).expand(v.size(0), self.encoder.max_role_count, self.encoder.max_role_count, cur_group.size(-1))
            cur_group2 = cur_group.unsqueeze(2).expand(v.size(0), self.encoder.max_role_count, self.encoder.max_role_count, cur_group.size(-1))
            cur_group1 = cur_group1.contiguous().view(-1, self.encoder.max_role_count * self.encoder.max_role_count, cur_group.size(-1))
            cur_group2 = cur_group2.contiguous().view(-1, self.encoder.max_role_count * self.encoder.max_role_count, cur_group.size(-1))

            concat_vec = torch.cat([cur_group1, cur_group2], 2).view(-1, cur_group.size(-1)*2)

            a = cur_group2.view(concat_vec.size(0), -1)

            added = torch.sigmoid(self.context_adder(concat_vec)) * a

            added = added.contiguous().view(v.size(0), self.encoder.max_role_count, self.encoder.max_role_count, cur_group.size(-1)).sum(2)

            withctx = added.contiguous().view(v.size(0) * self.encoder.max_role_count, -1)

            withctx_expand = withctx.expand(img.size(1), withctx.size(0), withctx.size(1))
            withctx_expand = withctx_expand.transpose(0,1)
            added_img = torch.cat([withctx_expand, img], -1)
            added_img = added_img.contiguous().view(-1, cur_group.size(-1)*3)
            added_img = torch.sigmoid(self.resize_ctx(added_img))
            added_img = added_img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, cur_group.size(-1)*2)

            img = added_img * img

            out = mfb_l2
            '''if prev is not None:
                out = prev + self.dropout(out)

            prev = out'''

        logits = self.classifier(out)

        loss = None
        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)
        if self.training:
            loss = self.calculate_loss(gt_verb, role_label_pred, labels)

        return role_label_pred, loss

    def forward(self, v_org, labels, gt_verb):

        img_features = self.convnet(v_org)
        #img_feat_flat = self.convnet.resnet.avgpool(img_features).squeeze()
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))


        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        n_heads = 4

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        q_emb_mul_head = q_emb.view(q_emb.size(0), n_heads, -1)
        q_emb_mul_head = q_emb_mul_head.contiguous().view(-1, q_emb_mul_head.size(-1))
        q_repr = self.q_net(q_emb_mul_head)
        prev = None

        '''prev = torch.zeros(batch_size * self.encoder.max_role_count, 1024)
        if torch.cuda.is_available():
            prev = prev.to(torch.device('cuda'))'''

        #vemb_list = [init_vemb]

        for i in range(2):


            img_mul_head = img.view(img.size(0), img.size(1),  n_heads, -1).transpose(1, 2)
            img_mul_head = img_mul_head.contiguous().view(-1, img_mul_head.size(2), img_mul_head.size(-1))

            #print('img q :', img_mul_head.size(), q_emb_mul_head.size())
            #attention

            att = self.v_att(img_mul_head, q_emb_mul_head)
            v_emb = (att * img_mul_head).sum(1) # [batch, v_dim]
            #vemb_list.append(v_emb)
            #v_emb = v_emb.contiguous().view(batch_size* self.encoder.max_role_count, -1)
            v_repr = self.v_net(v_emb)

            #composition

            mfb_iq_eltwise = torch.mul(q_repr, v_repr)

            mfb_iq_drop = self.Dropout_M(mfb_iq_eltwise)

            mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
            mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
            mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
            mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
            mfb_l2 = F.normalize(mfb_sign_sqrt)

            #contextualization

            cur_group = mfb_l2.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            #print('before att :', cur_group[1,:, :5])
            mask = self.encoder.get_adj_matrix_noself(gt_verb)

            if torch.cuda.is_available():
                mask = mask.to(torch.device('cuda'))

            selfatt_val= self.ctx_att(cur_group, cur_group, cur_group, mask=mask)

            #print('after att :', selfatt_val[1,:, :5])

            withctx = selfatt_val.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)

            withctx_expand = withctx.expand(img.size(1), withctx.size(0), withctx.size(1))
            withctx_expand = withctx_expand.transpose(0,1)
            added_img = torch.cat([withctx_expand, img], -1)
            added_img = added_img.contiguous().view(-1, cur_group.size(-1)*3)
            added_img = torch.sigmoid(self.resize_ctx(added_img))
            added_img = added_img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, cur_group.size(-1)*2)

            img = added_img * img

            #img = img * self.resize_ctx(withctx).unsqueeze(1)

            out = mfb_l2
            '''if prev is not None:
                out = prev + self.dropout(out)'''

            #prev = out

        logits = self.classifier(out)

        loss = None
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

    def calculate_loss1(self, role_label_pred, gt_labels):
        #loss = nn.KLDivLoss(reduction='sum')
        loss = nn.BCEWithLogitsLoss()
        final_loss = loss(role_label_pred, gt_labels) * role_label_pred.size(1)
        return final_loss

    def calculate_loss_new(self, batch_size, role_label_pred, gt_labels):
        #loss = nn.KLDivLoss(reduction='sum')
        loss = nn.BCEWithLogitsLoss(reduction='sum')
        role_leb_placewise = role_label_pred.view(batch_size, self.encoder.max_role_count, -1)
        gt_labels_placewise = gt_labels.view(batch_size, self.encoder.max_role_count, -1)
        role_loss = 0
        for j in range(0, self.encoder.max_role_count):
            n = 0
            curr_loss = 0
            for i in range(batch_size):
                if not (gt_labels_placewise[i][j][-1] > 0.0) :
                    n += 1
                    curr_loss += loss(role_leb_placewise[i][j], gt_labels_placewise[i][j])
            output = curr_loss / (n + 10e-8)
            role_loss += output
        return (role_loss / self.encoder.max_role_count)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                       dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class BaseModelGrid_Imsitu_RoleIter_Beam(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter, beam_size, upperlimit):
        super(BaseModelGrid_Imsitu_RoleIter_Beam, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter
        self.beam_size = beam_size
        self.upperlimit = upperlimit

    def forward(self, v, q, labels, gt_verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        losses = []
        prev_rep = None
        batch_size = v.size(0)
        for i in range(self.num_iter):

            img = v

            img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
            q = q.view(batch_size* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = joint_repr + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            if self.training:
                losses.append(self.calculate_loss(gt_verb, role_label_pred, labels))

            #label_idx = torch.max(role_label_pred,-1)[1]
            #do training with GT LABELS
            frame_idx = np.random.randint(3, size=1)
            label_idx = labels[:,frame_idx,:].squeeze()

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return role_label_pred, loss

    def forward_eval(self, v, q, gt_verb):
        '''
        rotate normally until num iter -1
        then come out of loop, take out of num iter -1
        get top k (beam) predictions of all roles
        make k x k combinations of label index, get their respective questions
        do same processing and get logits.

        for each role, for each q, get max logit and noun label. see which q gave the best logit.
        record that. select best nouns, only send those to scoring.

        '''
        prev_rep = None
        for i in range(self.num_iter - 1):

            img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

            q = q.view(v.size(0)* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = joint_repr + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)


            label_idx = torch.max(role_label_pred,-1)[1]

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

        # start beam
        sorted_idx = torch.sort(role_label_pred, -1, True)[1]
        sorted_role_labels = sorted_idx[:,:, :self.beam_size]
        # now need to create batchsize x (beam x beam) x 6 x 1 tensor with all combinations of labels starting from
        # top 1 of all roles ending with top beam of all roles

        all_role_combinations = self.get_role_combinations(sorted_role_labels)

        combo_size = all_role_combinations.size(1)

        beam_role_idx = None
        beam_role_value = None

        for k in range(0, combo_size):
            current_label_idx = all_role_combinations[:,k,:]
            #print('current size to make q :', current_label_idx.size())

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, current_label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

            img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

            q = q.view(v.size(0)* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            joint_repr = joint_repr + prev_rep

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            max_val, max_label_idx = torch.max(role_label_pred,-1)

            if k == 0:
                beam_role_idx = max_label_idx.unsqueeze(-1)
                beam_role_value = max_val.unsqueeze(-1)
            else:
                beam_role_idx = torch.cat((beam_role_idx.clone(), max_label_idx.unsqueeze(-1)), -1)
                beam_role_value = torch.cat((beam_role_value.clone(), max_val.unsqueeze(-1)), -1)


        best_noun_probs_idx = torch.max(beam_role_value,-1)[1]

        #linearize all dimentions
        noun_idx = best_noun_probs_idx.view(-1)
        role_idx_lin = beam_role_idx.view(-1, beam_role_idx.size(-1))

        selected_noun_labels = role_idx_lin.gather(1, noun_idx.unsqueeze(1))

        best_predictions = selected_noun_labels.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return best_predictions

    def forward_eval_dotproduct(self, v, q, gt_verb):
        '''
        rotate normally until num iter -1
        then come out of loop, take out of num iter -1
        get top k (beam) predictions of all roles
        make k x k combinations of label index, get their respective questions
        do same processing and get logits.

        for each role, for each q, get max logit and noun label. see which q gave the best logit.
        record that. select best nouns, only send those to scoring.

        '''
        prev_rep = None
        for i in range(self.num_iter - 1):

            img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

            q = q.view(v.size(0)* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = joint_repr + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)


            label_idx = torch.max(role_label_pred,-1)[1]

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

        # start beam
        sorted_idx = torch.sort(role_label_pred, -1, True)[1]
        sorted_role_labels = sorted_idx[:,:, :self.beam_size]
        # now need to create batchsize x (beam x beam) x 6 x 1 tensor with all combinations of labels starting from
        # top 1 of all roles ending with top beam of all roles

        all_role_combinations_tot = self.get_role_combinations(sorted_role_labels)
        all_role_combinations = all_role_combinations_tot[:, :self.upperlimit, :]


        combo_size = all_role_combinations.size(1)

        #get the noun weights of last layer of classifier
        noun_weights = self.classifier.main[-1].weight

        #further linearize combo to (batch x combo x 6), 1
        combo_1dim = all_role_combinations.contiguous().view(-1)
        selected_embeddings = torch.index_select(noun_weights, 0, combo_1dim)

        rearrage_embed = selected_embeddings.view(all_role_combinations.size(0), all_role_combinations.size(1),all_role_combinations.size(2), -1)

        tot_each_combo = torch.sum(rearrage_embed, 2)
        img_tot = torch.sum(v, 1)
        img_tot = img_tot.unsqueeze(1)
        img_match_combo = img_tot.expand(img_tot.size(0),all_role_combinations.size(1), img_tot.size(-1))

        '''dot_prod_all = tot_each_combo * img_match_combo
        print('dot_prod_all',dot_prod_all.size())'''

        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        cos_out = cos(tot_each_combo, img_match_combo)
        best_sim = torch.max(cos_out,-1)[1]

        best_combo = torch.gather(all_role_combinations, 1, best_sim.view(-1, 1).unsqueeze(2).repeat(1, 1, 6))

        best_label_idx = best_combo.squeeze()


        role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, best_label_idx)

        if torch.cuda.is_available():
            q = role_q_idx.to(torch.device('cuda'))

        img = v.expand(self.encoder.max_role_count,v.size(0), v.size(1), v.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, v.size(2))

        q = q.view(v.size(0)* self.encoder.max_role_count, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        joint_repr = joint_repr + prev_rep

        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return role_label_pred

    def get_role_combinations(self, sorted_role_labels):

        final_combo = None
        role = self.encoder.max_role_count
        value = self.beam_size
        tot = value ** role

        for i in range(role):
            role_num = i+1
            exp = value**(role-role_num)
            repeat = tot//(exp*value)

            current = sorted_role_labels[:,i]
            new_current = current
            if exp != 1:
                new_current = new_current.unsqueeze(-1)
                new_current = new_current.expand(new_current.size(0),-1, exp)
                new_current = new_current.contiguous().view(new_current.size(0),-1)

            if repeat != 1:
                new_current = new_current.unsqueeze(1)
                new_current = new_current.expand(new_current.size(0),repeat, -1)
                new_current = new_current.contiguous().view(new_current.size(0),-1)

            if i == 0:
                final_combo = new_current.unsqueeze(-1)
            else:
                final_combo = torch.cat((final_combo.clone(), new_current.unsqueeze(-1)), -1)

        return final_combo


    def forward_noq(self, v, verb):
        q = self.encoder.get_role_q_by_verb(verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_pred, _ = self.forward_eval(v, q, None, verb)

        return role_label_pred

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

class BaseModelGrid_Imsitu_RoleIter_IndiLoss(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter):
        super(BaseModelGrid_Imsitu_RoleIter_IndiLoss, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter

    def forward(self, v, q, labels, gt_verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        losses = []
        prev_rep = None
        batch_size = v.size(0)
        for i in range(self.num_iter):

            img = v

            img = img.expand(self.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
            img = img.transpose(0,1)
            img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))
            q = q.view(batch_size* self.encoder.max_role_count, -1)
            labels = labels.view(batch_size* self.encoder.max_role_count, -1)

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img, q_emb)
            v_emb = (att * img).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = joint_repr + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            if self.training:
                losses.append(self.calculate_loss(logits, labels))

            role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            label_idx = torch.max(role_label_pred,-1)[1]
            #for gt labels
            #frame_idx = np.random.randint(3, size=1)
            #label_idx = labels[:,frame_idx,:].squeeze()

            role_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = role_q_idx.to(torch.device('cuda'))

        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return role_label_pred, loss


    def calculate_loss(self, role_label_pred, gt_labels):

        loss = nn.BCEWithLogitsLoss()
        final_loss = loss(role_label_pred, gt_labels)
        return final_loss

class BaseModelGrid_Imsitu_Role4VerbNew(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter):
        super(BaseModelGrid_Imsitu_Role4VerbNew, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_iter = num_iter

    def forward(self, v, q):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        batch_size = v.size(0)

        img = v

        img = img.expand(2,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * 2, -1, v.size(2))
        q = q.view(batch_size* 2, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)

        role_label_pred = logits.contiguous().view(v.size(0), 2, -1)

        return role_label_pred

    def forward_reponly(self, v, q):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        batch_size = v.size(0)

        img = v

        img = img.expand(2,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * 2, -1, v.size(2))
        q = q.view(batch_size* 2, -1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        return joint_repr.contiguous().view(v.size(0), 2, -1)


    def forward_noq(self, v, verb=None):
        q = self.encoder.get_agent_place_roleqs(v.size(0), verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_pred = self.forward(v, q)

        return role_label_pred

    def forward_noq_reponly(self, v, verb=None):
        q = self.encoder.get_agent_place_roleqs(v.size(0), verb)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_rep = self.forward_reponly(v, q)
        role_label = self.forward(v, q)

        return role_label_rep, role_label

class BaseModelGrid_Imsitu_SingleRole(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_ans_classes):
        super(BaseModelGrid_Imsitu_SingleRole, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.num_ans_classes = num_ans_classes

    def forward(self, v, q):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        img = v
        q = q.squeeze()
        #print('inside model :', img.size(), q.size())

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        return logits

    def forward_noq(self, role, v, verb=None):
        if role == 'agent':
            q = self.encoder.get_agent_roleqs(v.size(0), verb)
        elif role == 'place':
            q = self.encoder.get_place_roleqs(v.size(0), verb)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        role_label_pred = self.forward(v, q)

        return role_label_pred

class BaseModelGrid_Imsitu_Verb(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder):
        super(BaseModelGrid_Imsitu_Verb, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder

    def forward(self, v, q):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """


        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        return logits

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbIter(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_VerbIter, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v.size(0)
        q = q.expand(batch_size, q.size(0))
        losses = []
        for i in range(self.num_iter):

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(v, q_emb)
            v_emb = (att * v).sum(1) # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr_prev = q_repr * v_repr
            logits = self.classifier(joint_repr_prev)

            if self.training:
                loss1 = self.calculate_loss(logits, gt_verbs)
                losses.append(loss1)

            sorted_idx = torch.sort(logits, 1, True)[1]
            verbs = sorted_idx[:,0]
            role_pred = self.role_module.forward_noq(v, verbs)
            label_idx = torch.max(role_pred,-1)[1]

            q = self.encoder.get_verbq_idx(verbs, label_idx)

            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter
        return logits, loss

    '''def forward(self, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss'''

    def forward_eval(self, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v.size(0)
        q = q.expand(batch_size, q.size(0))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        sorted_idx = torch.sort(logits, 1, True)[1]
        verbs = sorted_idx[:,0]
        role_pred = self.role_module.forward_noq(v, verbs)
        label_idx = torch.max(role_pred,-1)[1]

        q = self.encoder.get_verbq_idx(verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None


        return logits, loss


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbRoleVerb_General(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_VerbRoleVerb_General, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v_verb.size(0)
        q = q.expand(batch_size, q.size(0))
        losses = []
        for i in range(self.num_iter):

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(v_verb, q_emb)
            v_emb = (att * v_verb).sum(1) # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr_prev = q_repr * v_repr
            logits = self.classifier(joint_repr_prev)

            if self.training:
                loss1 = self.calculate_loss(logits, gt_verbs)
                losses.append(loss1)

            sorted_idx = torch.sort(logits, 1, True)[1]
            verbs = sorted_idx[:,0]
            role_pred = self.role_module.forward_noq(v_role, verbs)
            label_idx = torch.max(role_pred,-1)[1]

            q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)

            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter
        return logits, loss

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbRoleVerb_General_GTq_Train(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_VerbRoleVerb_General_GTq_Train, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(img_id, gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss

    def forward_eval(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v_verb.size(0)
        q = q.expand(batch_size, q.size(0))
        losses = []
        for i in range(self.num_iter):

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(v_verb, q_emb)
            v_emb = (att * v_verb).sum(1) # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr_prev = q_repr * v_repr
            logits = self.classifier(joint_repr_prev)

            sorted_idx = torch.sort(logits, 1, True)[1]
            verbs = sorted_idx[:,0]
            role_pred = self.role_module.forward_noq(v_role, verbs)
            label_idx = torch.max(role_pred,-1)[1]

            q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)

            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))


        return logits

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbRoleVerb_General_Corrected_Vq(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_VerbRoleVerb_General_Corrected_Vq, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v_verb.size(0)
        q = q.expand(batch_size, q.size(0))
        losses = []
        for i in range(self.num_iter):

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(v_verb, q_emb)
            v_emb = (att * v_verb).sum(1) # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr_prev = q_repr * v_repr
            logits = self.classifier(joint_repr_prev)

            if self.training:
                loss1 = self.calculate_loss(logits, gt_verbs)
                losses.append(loss1)

            sorted_idx = torch.sort(logits, 1, True)[1]
            verbs = sorted_idx[:,0]
            role_pred = self.role_module.forward_noq(v_role, verbs)
            label_idx = torch.max(role_pred,-1)[1]

            q = self.encoder.get_verbq_with_agentplace_with_verb(img_id, batch_size, label_idx, verbs)

            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter
        return logits, loss

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_RoleVerb_General(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_RoleVerb_General, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)
        label_idx = torch.max(role_pred,-1)[1]

        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss1 = self.calculate_loss(logits, gt_verbs)

        return logits, loss1

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_RoleVerb_General_Ctxcls(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier,q_emb_ctx,
                 v_att_ctx, v_net_ctx, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_RoleVerb_General_Ctxcls, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.q_emb_ctx = q_emb_ctx
        self.v_att_ctx = v_att_ctx
        self.v_net_ctx = v_net_ctx
        self.transform = nn.Linear(2048, 1024)
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter
        self.dropout = nn.Dropout(0.3)

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred_rep, role_pred = self.role_module.forward_noq_reponly(v_role)
        ctx_combined = self.q_emb_ctx(role_pred_rep)
        #ctx_combined = torch.sum(role_pred_rep, 1)
        att_ctx = self.v_att_ctx(v_verb, ctx_combined)
        v_emb_ctx = (att_ctx * v_verb).sum(1)
        v_repr_ctx = self.v_net_ctx(v_emb_ctx)
        q_repr_ctx = ctx_combined


        label_idx = torch.max(role_pred,-1)[1]

        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        joint_repr_ctx = q_repr_ctx * v_repr_ctx
        joint_repr_tot = self.dropout(self.transform(torch.cat([joint_repr_prev, joint_repr_ctx], -1)))
        logits = self.classifier(joint_repr_tot)

        loss1 = self.calculate_loss(logits, gt_verbs)

        return logits, loss1

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_RoleVerbIter_General_With_CNN(nn.Module):
    def __init__(self, convnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_RoleVerbIter_General_With_CNN, self).__init__()
        self.convnet = convnet
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter
        self.dropout = nn.Dropout(0.3)

    def forward_gt(self, img_id, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        img = v

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(img_id, gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss

    def forward(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_pred = self.role_module.forward_noq(v)
        for i in range(self.num_iter):


            label_idx = torch.max(role_pred,-1)[1]
            q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img_org, q_emb)
            v_emb = (att * img_org).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            if self.training:
                losses.append(self.calculate_loss(logits, gt_verbs))

            verb_idx = torch.max(logits,-1)[1]

            role_pred = self.role_module.forward_noq(v, verb_idx)


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return logits, loss

    def forward_eval(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_pred = self.role_module.forward_noq(v)
        for i in range(self.num_iter):


            label_idx = torch.max(role_pred,-1)[1]
            q, agent_names, place_names = self.encoder.get_verbq_with_agentplace_eval(img_id, batch_size, label_idx)
            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img_org, q_emb)
            v_emb = (att * img_org).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr

            logits = self.classifier(joint_repr)

            if self.training:
                losses.append(self.calculate_loss(logits, gt_verbs))

            verb_idx = torch.max(logits,-1)[1]

            role_pred = self.role_module.forward_noq(v, verb_idx)


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return logits, loss, agent_names, place_names

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        return final_loss

class BaseModelGrid_Imsitu_RoleVerbIter_General_With_CNN_ExtCtx(nn.Module):
    def __init__(self, convnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_RoleVerbIter_General_With_CNN_ExtCtx, self).__init__()
        self.convnet = convnet
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter
        self.dropout = nn.Dropout(0.3)
        self.resize_img_flat = nn.Linear(2048, 1024)

        #self.image_constructor = MLP(1024, 1024, 1024, num_layers=2, dropout_p=0.2)
        self.l2_criterion = nn.MSELoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward_gt(self, img_id, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        img_features = self.convnet(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        v = img_org.permute(0, 2, 1)

        img = v

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(img_id, gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss

    def forward_prev(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)
        partial_combo_stack = None
        for i in range(self.num_iter):

            role_rep_combo = torch.sum(role_rep, 1)
            ext_ctx = img_feat_flat * role_rep_combo
            label_idx = torch.max(role_pred,-1)[1]
            q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img_org, q_emb)
            v_emb = (att * img_org).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)


            joint_repr = q_repr * v_repr
            '''if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr'''

            rep = joint_repr + ext_ctx

            if i == 0:
                combo_rep = rep
                partial_combo_stack = rep.unsqueeze(1)
            else :
                #multi_rep = rep.unsqueeze(1).expand(partial_combo_stack.size(0),partial_combo_stack.size(1), partial_combo_stack.size(-1))
                #combo = torch.sum(self.tanh(self.rep_ctx_project(partial_combo_stack - multi_rep)),1) + rep
                #combo_rep = combo



                partial_combo_stack = torch.cat([partial_combo_stack.clone(), rep.unsqueeze(1)], 1)
                #combo_weights_q = self.combo_att_q(partial_combo_stack, q_repr)
                attn4q = partial_combo_stack * q_repr.unsqueeze(1)
                qattn = self.combo_att_q(attn4q).squeeze(2)
                qattn = F.softmax(qattn, 1).unsqueeze(2)
                combo_rep_q = (qattn * partial_combo_stack).sum(1)

                attn4img = partial_combo_stack * img_feat_flat.unsqueeze(1)
                iattn = self.combo_att_img(attn4img).squeeze(2)
                iattn = F.softmax(iattn, 1).unsqueeze(2)
                combo_rep_img = (iattn * partial_combo_stack).sum(1)

                #combo_weights_img = self.combo_att_img(partial_combo_stack, img_feat_flat)
                #combo_rep_img = (combo_weights_img * partial_combo_stack).sum(1)
                #v_repr_combo = torch.sum(partial_ans_stack, 1)

                gate = self.sigmoid(q_repr * img_feat_flat)

                combo_rep = gate * combo_rep_q + (1-gate) * combo_rep_img

            logits = self.classifier(combo_rep)

            if self.training:
                losses.append(self.calculate_loss(logits, gt_verbs))

            verb_idx = torch.max(logits,-1)[1]

            role_rep, role_pred = self.role_module.forward_noq_reponly(v, verb_idx)


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return logits, loss

    def forward_try(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)
        partial_combo_stack = torch.zeros(batch_size,1, 1024)
        if torch.cuda.is_available():
            partial_combo_stack = partial_combo_stack.to(torch.device('cuda'))

        #self.partial_ans0.expand(batch_size, 1, 1024)
        for i in range(self.num_iter):

            role_rep_combo = torch.sum(role_rep, 1)
            ext_ctx = img_feat_flat * role_rep_combo
            label_idx = torch.max(role_pred,-1)[1]
            q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img_org, q_emb)
            v_emb = (att * img_org).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)


            joint_repr = q_repr * v_repr
            '''if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr'''

            rep = joint_repr + ext_ctx

            attn4img = partial_combo_stack * img_feat_flat.unsqueeze(1)
            iattn = self.combo_att_img(attn4img).squeeze(2)
            iattn = F.softmax(iattn, 1).unsqueeze(2)
            combo_rep_img = (iattn * partial_combo_stack).sum(1)

            gate = self.sigmoid(q_repr * img_feat_flat)
            combo_rep = gate * rep + (1-gate) * combo_rep_img

            logits = self.classifier(combo_rep)

            if self.training:
                losses.append(self.calculate_loss(logits, gt_verbs))

            verb_idx = torch.max(logits,-1)[1]

            role_rep, role_pred = self.role_module.forward_noq_reponly(v, verb_idx)


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return logits, loss

    def forward_rolectx(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)

        #self.partial_ans0.expand(batch_size, 1, 1024)
        for i in range(self.num_iter):

            role_rep_combo = torch.sum(role_rep, 1)

            ext_ctx = img_feat_flat * role_rep_combo

            label_idx = torch.max(role_pred,-1)[1]
            q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img_org, q_emb)
            v_emb = (att * img_org).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)


            joint_repr = q_repr * v_repr
            '''if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr'''

            #gate = self.sigmoid(q_repr * img_feat_flat)
            #combo_rep = gate * joint_repr + (1-gate) * ext_ctx

            combo_rep = joint_repr + ext_ctx

            logits = self.classifier(combo_rep)

            if self.training:
                losses.append(self.calculate_loss(logits, gt_verbs))

            verb_idx = torch.max(logits,-1)[1]

            role_rep, role_pred = self.role_module.forward_noq_reponly(v, verb_idx)


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return logits, loss

    def forward_muinfo(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)

        role_rep_combo = torch.sum(role_rep, 1)
        ext_ctx = img_feat_flat * role_rep_combo

        label_idx = torch.max(role_pred,-1)[1]
        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img_org, q_emb)
        v_emb = (att * img_org).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = q_repr * v_repr

        combo_rep = joint_repr + ext_ctx

        logits = self.classifier(combo_rep)

        recon_img = self.img_reconstructor(combo_rep)


        loss = None
        if self.training:
            cros_entropy = self.calculate_loss(logits, gt_verbs)
            l2 = self.l2_criterion(recon_img, img_feat_flat)
            #print(cros_entropy, l2)
            loss = cros_entropy + l2

        return logits, loss

    def forward_lkdiv(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)

        role_rep_combo = torch.sum(role_rep, 1)
        ext_ctx = img_feat_flat * role_rep_combo

        label_idx = torch.max(role_pred,-1)[1]
        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img_org, q_emb)
        v_emb = (att * img_org).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = q_repr * v_repr

        combo_rep = joint_repr + ext_ctx

        logits = self.classifier(combo_rep)

        mu_wrongq = self.mu_answer_encoder(combo_rep)
        var_wrongq = self.logvar_answer_encoder(combo_rep)


        #============================ gold q calc ==============================
        frame_idx = np.random.randint(3, size=1)
        label_idx_gold = labels[:,frame_idx,:].squeeze()
        q_gold = self.encoder.get_verbq_idx(img_id, gt_verbs, label_idx_gold)

        if torch.cuda.is_available():
            q_gold = q_gold.to(torch.device('cuda'))

        w_emb_gold = self.w_emb(q_gold)
        q_emb_gold = self.q_emb(w_emb_gold) # [batch, q_dim]

        att_gold = self.v_att(img_org, q_emb_gold)
        v_emb_gold = (att_gold * img_org).sum(1) # [batch, v_dim]

        q_repr_gold = self.q_net(q_emb_gold)
        v_repr_gold = self.v_net(v_emb_gold)

        joint_repr_gold = q_repr_gold * v_repr_gold

        mu_golq = self.mu_answer_encoder(joint_repr_gold)
        var_goldq = self.logvar_answer_encoder(joint_repr_gold)





        loss = None
        if self.training:
            cros_entropy = self.calculate_loss(logits, gt_verbs)
            kl_loss_wrongq = self.gaussian_KL_loss(mu_wrongq, var_wrongq)
            kl_loss_goldq = self.gaussian_KL_loss(mu_golq, var_goldq)
            kl_div = self.compute_two_gaussian_loss(mu_golq, var_goldq, mu_wrongq, var_wrongq)

            #print(cros_entropy, kl_loss_wrongq, kl_loss_goldq, kl_div)
            loss = cros_entropy + kl_loss_wrongq + kl_loss_goldq + kl_div

        return logits, loss

    def forward_multihead(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        #img_feat_flat = self.convnet.resnet.avgpool(img_features)
        #img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        multi_headed_img = img_org.view(img_org.size(0), img_org.size(1), self.n_heads, img_org.size(2)//self.n_heads)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)

        role_rep_combo = torch.sum(role_rep, 1)
        role_resized = self.context_shaper_mul(role_rep_combo)
        ctx_multiheaded_mul = role_resized.view(role_resized.size(0), 1, self.n_heads, role_resized.size(1)//self.n_heads)
        contextualized_multiheaded_img = multi_headed_img * ctx_multiheaded_mul

        contextualized_img = self.non_linear_combinator(
            contextualized_multiheaded_img.view(img_org.size(0), -1, img_org.size(2)))

        soft_vqa_ans = self.avg_pool(contextualized_img).squeeze()

        updated_img = contextualized_img

        label_idx = torch.max(role_pred,-1)[1]
        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(updated_img, q_emb)
        v_emb = (att * updated_img).sum(1) # [batch, v_dim]
        #v_emb = self.avg_pool_role(att * updated_img).squeeze()

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        hard_vqa_ans = q_repr * v_repr + soft_vqa_ans


        combo_rep = hard_vqa_ans


        logits = self.classifier(combo_rep)

        loss = None

        if self.training:
            cros_entropy = self.calculate_loss(logits, gt_verbs)
            #print(cros_entropy, l2)
            loss = cros_entropy

        return logits, loss

    def forward_eval1(self, img_id, v, gt_verbs, labels): #change this according to the current foreward method

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)

        #self.partial_ans0.expand(batch_size, 1, 1024)
        for i in range(self.num_iter):

            role_rep_combo = torch.sum(role_rep, 1)
            ext_ctx = img_feat_flat * role_rep_combo
            label_idx = torch.max(role_pred,-1)[1]

            label_sfmx, label_id = torch.sort(role_pred, -1, True)
            #print(label_idx.size(),label_sfmx.size(), label_id.size(), role_pred.size() )
            label_sfmx_10 = label_sfmx[:,:,:10]
            label_id_10 = label_id[:,:,:10]
            #print(label_sfmx_10[1,0], label_id_10[1,0])


            q, agents, places, agent_10, place_10 = self.encoder.get_verbq_with_agentplace_eval(img_id, batch_size, label_idx, label_sfmx_10, label_id_10)
            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(img_org, q_emb)
            v_emb = (att * img_org).sum(1) # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)


            joint_repr = q_repr * v_repr
            '''if i != 0:
                joint_repr = self.dropout(joint_repr) + prev_rep
            prev_rep = joint_repr'''

            combo_rep = joint_repr + ext_ctx

            logits = self.classifier(combo_rep)

            if self.training:
                losses.append(self.calculate_loss(logits, gt_verbs))

            verb_idx = torch.max(logits,-1)[1]

            role_rep, role_pred = self.role_module.forward_noq_reponly(v, verb_idx)


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter


        return logits, loss, agents, places, agent_10, place_10

    def forward_ctx(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)

        role_rep_combo = torch.sum(role_rep, 1)

        ext_ctx = img_feat_flat * role_rep_combo

        label_idx = torch.max(role_pred,-1)[1]
        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img_org, q_emb)
        v_emb = (att * img_org).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = q_repr * v_repr

        combo_rep = joint_repr + ext_ctx
        #combo_rep = joint_repr

        logits = self.classifier(combo_rep)

        loss = None
        if self.training:
            cros_entropy = self.calculate_loss(logits, gt_verbs)
            #print(cros_entropy, l2)
            loss = cros_entropy

        return logits, loss

    def forward(self, img_id, v, gt_verbs, labels):

        img_features = self.convnet(v)
        img_feat_flat = self.convnet.resnet.avgpool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, n_channel, -1)
        img_org = img_org.permute(0, 2, 1)

        losses = []
        prev_rep = None
        batch_size = v.size(0)
        role_rep, role_pred = self.role_module.forward_noq_reponly(v)
        role_rep_img = img_feat_flat.unsqueeze(1) * role_rep

        role_rep_combo = torch.sum(role_rep_img, 1)

        #ext_ctx = img_feat_flat * role_rep_combo
        ext_ctx = role_rep_combo

        label_idx = torch.max(role_pred,-1)[1]
        q, grounded_info = self.encoder.get_verbq_with_agentplace_with_grounded_info(img_id, batch_size, label_idx, role_rep_img)
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))
            grounded_info = grounded_info.to(torch.device('cuda'))

        w_emb = self.w_emb(q) + grounded_info
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img_org, q_emb)
        v_emb = (att * img_org).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = q_repr * v_repr

        combo_rep = joint_repr + ext_ctx

        logits = self.classifier(combo_rep)

        loss = None
        if self.training:
            cros_entropy = self.calculate_loss(logits, gt_verbs)
            #print(cros_entropy, l2)
            loss = cros_entropy

        return logits, loss

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        return final_loss

    def compute_two_gaussian_loss(self, mu1, logvar1, mu2, logvar2):
        """Computes the KL loss between the embedding attained from the answers
        and the categories.
        KL divergence between two gaussians:
            log(sigma_2/sigma_1) + (sigma_2^2 + (mu_1 - mu_2)^2)/(2sigma_1^2) - 0.5
        Args:
            mu1: Means from first space.
            logvar1: Log variances from first space.
            mu2: Means from second space.
            logvar2: Means from second space.
        """
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp() + 1e-8))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1)
        return kl / (mu1.size(0) + 1e-8)

    def gaussian_KL_loss(self, mus, logvars, eps=1e-8):
        """Calculates KL distance of mus and logvars from unit normal.
        Args:
            mus: Tensor of means predicted by the encoder.
            logvars: Tensor of log vars predicted by the encoder.
        Returns:
            KL loss between mus and logvars and the normal unit gaussian.
        """
        KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
        kl_loss = KLD/(mus.size(0) + eps)
        """
        if kl_loss > 100:
            print kl_loss
            print KLD
            print mus.min(), mus.max()
            print logvars.min(), logvars.max()
            1/0
        """
        return kl_loss


class BaseModelGrid_Imsitu_RoleVerb_General_GTq_Train(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_RoleVerb_General_GTq_Train, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(img_id, gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss

    def forward_eval(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)
        label_idx = torch.max(role_pred,-1)[1]

        #get a beam
        sorted_val, sorted_idx = torch.sort(role_pred, -1, True)
        top5 = sorted_idx[:,:,:10]

        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, label_idx, top5)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        return logits

    def forward_eval_gttemplte_predlabel(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)
        label_idx = torch.max(role_pred,-1)[1]

        frame_idx = np.random.randint(3, size=1)
        gt_label_idx = labels[:,frame_idx,:].squeeze()

        q = self.encoder.get_verbq_goldtemplate_predlabels(img_id, gt_verbs, gt_label_idx, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        return logits

    def forward_eval_gttemplte_predagent(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)
        label_idx = torch.max(role_pred,-1)[1]

        frame_idx = np.random.randint(3, size=1)
        gt_label_idx = labels[:,frame_idx,:].squeeze()

        q = self.encoder.get_verbq_goldtemplate_predagent(img_id, gt_verbs, gt_label_idx, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        return logits

    def forward_eval_gttemplte_predplace(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)
        label_idx = torch.max(role_pred,-1)[1]

        frame_idx = np.random.randint(3, size=1)
        gt_label_idx = labels[:,frame_idx,:].squeeze()

        q = self.encoder.get_verbq_goldtemplate_predplace(img_id, gt_verbs, gt_label_idx, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        return logits

    def forward_eval_predtemplate_gtlabels(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)
        label_idx = torch.max(role_pred,-1)[1]

        frame_idx = np.random.randint(3, size=1)
        gt_label_idx = labels[:,frame_idx,:].squeeze()

        q = self.encoder.get_verbq_predtemplate_goldlabels(img_id, gt_verbs, gt_label_idx, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        return logits

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_RoleVerb_General_GTq_Train_Beam(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter, beam_size):
        super(BaseModelGrid_Imsitu_RoleVerb_General_GTq_Train_Beam, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter
        self.beam_size = beam_size

    def forward(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(img_id, gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss

    def forward_eval(self, img_id, v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        role_pred = self.role_module.forward_noq(v_role)

        #get a beam


        sorted_idx = torch.sort(role_pred, -1, True)[1]
        sorted_role_labels = sorted_idx[:,:, :self.beam_size]
        # now need to create batchsize x (beam x beam) x 6 x 1 tensor with all combinations of labels starting from
        # top 1 of all roles ending with top beam of all roles

        all_role_combinations_tot = self.get_role_combinations(sorted_role_labels)
        all_role_combinations = all_role_combinations_tot


        #get the noun weights of last layer of classifier
        noun_weights = self.role_module.classifier.main[-1].weight

        #further linearize combo to (batch x combo x 6), 1
        combo_1dim = all_role_combinations.contiguous().view(-1)
        selected_embeddings = torch.index_select(noun_weights, 0, combo_1dim)

        rearrage_embed = selected_embeddings.view(all_role_combinations.size(0), all_role_combinations.size(1),all_role_combinations.size(2), -1)

        tot_each_combo = torch.sum(rearrage_embed, 2)
        img_tot = torch.sum(v_verb, 1)
        img_tot = img_tot.unsqueeze(1)
        img_match_combo = img_tot.expand(img_tot.size(0),all_role_combinations.size(1), img_tot.size(-1))

        '''dot_prod_all = tot_each_combo * img_match_combo
        print('dot_prod_all',dot_prod_all.size())'''

        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        cos_out = cos(tot_each_combo, img_match_combo)
        best_sim = torch.max(cos_out,-1)[1]

        best_combo = torch.gather(all_role_combinations, 1, best_sim.view(-1, 1).unsqueeze(2).repeat(1, 1, all_role_combinations.size(-1)))

        best_label_idx = best_combo.squeeze()

        q = self.encoder.get_verbq_with_agentplace(img_id, batch_size, best_label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        return logits, None

    def get_role_combinations(self, sorted_role_labels):
        final_combo = None
        role = sorted_role_labels.size(1)
        value = self.beam_size
        tot = value ** role

        for i in range(role):
            role_num = i+1
            exp = value**(role-role_num)
            repeat = tot//(exp*value)

            current = sorted_role_labels[:,i]
            new_current = current
            if exp != 1:
                new_current = new_current.unsqueeze(-1)
                new_current = new_current.expand(new_current.size(0),-1, exp)
                new_current = new_current.contiguous().view(new_current.size(0),-1)

            if repeat != 1:
                new_current = new_current.unsqueeze(1)
                new_current = new_current.expand(new_current.size(0),repeat, -1)
                new_current = new_current.contiguous().view(new_current.size(0),-1)

            if i == 0:
                final_combo = new_current.unsqueeze(-1)
            else:
                final_combo = torch.cat((final_combo.clone(), new_current.unsqueeze(-1)), -1)

        return final_combo

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbRoleVerb_Special(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, agent_module, place_module,  num_iter):
        super(BaseModelGrid_Imsitu_VerbRoleVerb_Special, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.agent_module = agent_module
        self.place_module = place_module
        self.num_iter = num_iter

    def forward(self, img_id,v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v_verb.size(0)
        q = q.expand(batch_size, q.size(0))
        losses = []
        for i in range(self.num_iter):

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(v_verb, q_emb)
            v_emb = (att * v_verb).sum(1) # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr_prev = q_repr * v_repr
            logits = self.classifier(joint_repr_prev)

            if self.training:
                loss1 = self.calculate_loss(logits, gt_verbs)
                losses.append(loss1)

            sorted_idx = torch.sort(logits, 1, True)[1]
            verbs = sorted_idx[:,0]
            agent_pred = self.agent_module.forward_noq("agent", v_role, verbs)
            agent_idx = torch.max(agent_pred,-1)[1]
            place_pred = self.place_module.forward_noq("place", v_role, verbs)
            place_idx = torch.max(place_pred,-1)[1]

            q = self.encoder.get_verbq_with_agentplace_special(img_id,batch_size, agent_idx, place_idx)

            if torch.cuda.is_available():
                q = q.to(torch.device('cuda'))


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter
        return logits, loss

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_RoleVerb_Special(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, agent_module, place_module,  num_iter):
        super(BaseModelGrid_Imsitu_RoleVerb_Special, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.agent_module = agent_module
        self.place_module = place_module
        self.num_iter = num_iter

    def forward(self, img_id,v_verb, v_role, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        batch_size = v_verb.size(0)
        agent_pred = self.agent_module.forward_noq("agent", v_role)
        agent_idx = torch.max(agent_pred,-1)[1]
        place_pred = self.place_module.forward_noq("place", v_role)
        place_idx = torch.max(place_pred,-1)[1]

        q = self.encoder.get_verbq_with_agentplace_special(img_id,batch_size, agent_idx, place_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v_verb, q_emb)
        v_emb = (att * v_verb).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss1 = self.calculate_loss(logits, gt_verbs)

        return logits, loss1

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbIterCNN(nn.Module):
    def __init__(self, cnn, conv_exp, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_VerbIterCNN, self).__init__()
        self.conv = cnn
        self.conv_exp = conv_exp
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, image, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q\
        #ENCODE THE IMAGE USING CNN on the fly
        conv_out = self.conv(image)
        v = self.conv_exp(conv_out)
        batch_size, n_channel, conv_h, conv_w = v.size()
        v = v.view(batch_size, n_channel, -1)
        v = v.permute(0, 2, 1)

        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v.size(0)
        q = q.expand(batch_size, q.size(0))
        losses = []
        for i in range(self.num_iter):

            w_emb = self.w_emb(q)
            q_emb = self.q_emb(w_emb) # [batch, q_dim]

            att = self.v_att(v, q_emb)
            v_emb = (att * v).sum(1) # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr_prev = q_repr * v_repr
            logits = self.classifier(joint_repr_prev)

            if self.training:
                loss1 = self.calculate_loss(logits, gt_verbs)
                losses.append(loss1)

            if self.num_iter > 1:
                sorted_idx = torch.sort(logits, 1, True)[1]
                verbs = sorted_idx[:,0]
                role_pred = self.role_module.forward_noq(v, verbs)
                label_idx = torch.max(role_pred,-1)[1]

                q = self.encoder.get_verbq_idx(verbs, label_idx)

                if torch.cuda.is_available():
                    q = q.to(torch.device('cuda'))


        loss = None
        if self.training:
            loss_all = torch.stack(losses,0)
            loss = torch.sum(loss_all, 0)/self.num_iter
        return logits, loss

    '''def forward(self, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q

        frame_idx = np.random.randint(3, size=1)
        label_idx = labels[:,frame_idx,:].squeeze()
        q = self.encoder.get_verbq_idx(gt_verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None

        if self.training:
            loss = self.calculate_loss(logits, gt_verbs)

        return logits, loss'''

    def forward_eval(self, v, gt_verbs, labels):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q
        q = self.encoder.get_generalq()
        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        batch_size = v.size(0)
        q = q.expand(batch_size, q.size(0))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        sorted_idx = torch.sort(logits, 1, True)[1]
        verbs = sorted_idx[:,0]
        role_pred = self.role_module.forward_noq(v, verbs)
        label_idx = torch.max(role_pred,-1)[1]

        q = self.encoder.get_verbq_idx(verbs, label_idx)

        if torch.cuda.is_available():
            q = q.to(torch.device('cuda'))

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr_prev = q_repr * v_repr
        logits = self.classifier(joint_repr_prev)

        loss = None


        return logits, loss


    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils_imsitu.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BaseModelGrid_Imsitu_VerbIter_Resnet_FeatExtract(nn.Module):
    def __init__(self, cnn, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter):
        super(BaseModelGrid_Imsitu_VerbIter_Resnet_FeatExtract, self).__init__()
        self.conv = cnn
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.role_module = role_module
        self.num_iter = num_iter

    def forward(self, image):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #iter 0 with general q\
        #ENCODE THE IMAGE USING CNN on the fly
        img_features = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        v = img_features.view(batch_size, n_channel, -1)
        v = v.permute(0, 2, 1)

        return v


class BaseModelGrid_Imsitu_Verb_Role_Joint_Eval(nn.Module):
    def __init__(self, encoder, verb_module, role_module):
        super(BaseModelGrid_Imsitu_Verb_Role_Joint_Eval, self).__init__()

        self.encoder = encoder
        self.verb_module = verb_module
        self.role_module = role_module


    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward(self, img_id, v, gt_verbs, labels, topk=5):

        role_pred_topk = None

        verb_pred, _ = self.verb_module(img_id, v, gt_verbs, labels)

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        verbs = sorted_idx[:,:topk]

        for k in range(0,topk):
            questions = self.encoder.get_role_nl_questions_batch(verbs[:,k])
            if torch.cuda.is_available():
                questions = questions.to(torch.device('cuda'))

            role_pred, _ = self.role_module(v, questions,labels, verbs[:,k])

            if k == 0:
                idx = torch.max(role_pred,-1)[1]
                role_pred_topk = idx
            else:
                idx = torch.max(role_pred,-1)[1]
                role_pred_topk = torch.cat((role_pred_topk.clone(), idx), 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return verbs, role_pred_topk

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_baseline0grid(dataset, num_hid):
    conv_net = resnet_152_features()
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModelGrid(conv_net, w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_baseline0grid_imsitu(dataset, num_hid, num_ans_classes, encoder):
    print('words count :', dataset.dictionary.ntoken)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder)

def build_baseline0grid_imsitu_agent(dataset, num_hid, num_ans_classes, encoder):
    print('words count :', dataset.dictionary.ntoken)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_Agent( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_ans_classes)

def build_baseline0grid_imsitu_singlerole(dataset, num_hid, num_ans_classes, encoder):
    print('words count :', encoder.roleq_dict.ntoken)
    w_emb = WordEmbedding(encoder.roleq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_SingleRole( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_ans_classes)

def build_baseline0grid_imsitu_roleiter(dataset, num_hid, num_ans_classes, encoder, num_iter):
    print('words count :', dataset.dictionary.ntoken)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter(w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter)

def build_baseline0grid_imsitu_roleiter_with_cnn(dataset, num_hid, num_ans_classes, encoder, num_iter):
    print('words count :', dataset.dictionary.ntoken)
    covnet = resnet_modified_medium()
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter_With_CNN(covnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter)

def build_baseline0grid_imsitu_roleiter_with_cnn_extctx(dataset, num_hid, num_ans_classes, encoder, num_iter):
    print('words count :', dataset.dictionary.ntoken)
    n_heads = 4
    covnet = resnet_modified_medium()
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048//n_heads, q_emb.num_hid//n_heads, num_hid)
    q_net = FCNet([num_hid//n_heads, num_hid ])
    v_net = FCNet([2048//n_heads, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter_With_CNN_EXTCTX(covnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter)

def build_baseline0grid_imsitu_roleiter_with_cnn_newmodel(num_hid, n_roles, n_verbs, num_ans_classes, encoder, num_iter):
    #print('words count :', dataset.dictionary.ntoken)
    n_heads = 4
    covnet = resnet_modified_medium()
    role_emb = nn.Embedding(n_roles+1, 300, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, 300)
    query_composer = FCNet([600, 1024])
    v_att = Attention(2048//n_heads, 1024//n_heads, num_hid)
    q_net = FCNet([num_hid//n_heads, num_hid ])
    v_net = FCNet([2048//n_heads, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter_With_CNN_NewModel(covnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, classifier, encoder, num_iter)

def build_baseline0grid_imsitu_roleiter_beam(dataset, num_hid, num_ans_classes, encoder, num_iter, beam_size, upperlimit):
    print('words count :', dataset.dictionary.ntoken)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter_Beam(w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter,
                                              beam_size, upperlimit)

def build_baseline0grid_imsitu_roleiter_indiloss(dataset, num_hid, num_ans_classes, encoder, num_iter):
    print('words count :', dataset.dictionary.ntoken)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter_IndiLoss(w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter)

def build_baseline0grid_imsitu4verb(dataset, num_hid, num_ans_classes, encoder, num_iter):
    print('words count :', encoder.roleq_dict.ntoken)
    w_emb = WordEmbedding(encoder.roleq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_Role4VerbNew( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter)

def build_baseline0grid_imsitu_verb(dataset, num_hid, num_ans_classes, encoder):
    print('words count :', dataset.dictionary.ntoken)
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_Verb( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder)

def build_baseline0grid_imsitu_verbiter(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbIter( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_verbroleverb_general(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbRoleVerb_General( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_verbroleverb_general_gtq_train(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbRoleVerb_General_GTq_Train( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_verbroleverb_general_corrected_vq(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbRoleVerb_General_Corrected_Vq( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_roleverb_general(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_RoleVerb_General( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_roleverb_general_ctxcls(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    q_emb_ctx = QuestionEmbedding(num_hid, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    v_att_ctx = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    v_net_ctx = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_RoleVerb_General_Ctxcls( w_emb, q_emb, v_att, q_net, v_net, classifier,q_emb_ctx,
                                                         v_att_ctx, v_net_ctx, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_roleverb_general_with_cnn(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.dictionary.ntoken)
    covnet = resnet_modified_medium()
    w_emb = WordEmbedding(encoder.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_RoleVerbIter_General_With_CNN(covnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_roleverb_general_with_cnn_extctx(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.dictionary.ntoken)
    covnet = resnet_modified_medium()
    w_emb = WordEmbedding(encoder.dictionary.ntoken, num_hid, 0.0)
    q_emb = QuestionEmbedding(num_hid, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_RoleVerbIter_General_With_CNN_ExtCtx(covnet, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_roleverb_general_gtq_train(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_RoleVerb_General_GTq_Train( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_roleverb_general_gtq_train_beam(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter, beam_size):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_RoleVerb_General_GTq_Train_Beam( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter, beam_size)

def build_baseline0grid_imsitu_verbroleverb_special(dataset, num_hid, num_ans_classes, encoder, agent_module, place_module,  num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    agent_module = agent_module
    place_module = place_module
    return BaseModelGrid_Imsitu_VerbRoleVerb_Special( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, agent_module, place_module,  num_iter)

def build_baseline0grid_imsitu_roleverb_special(dataset, num_hid, num_ans_classes, encoder, agent_module, place_module,  num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    agent_module = agent_module
    place_module = place_module
    return BaseModelGrid_Imsitu_RoleVerb_Special( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, agent_module, place_module,  num_iter)


def build_baseline0grid_imsitu_verbiterCNN(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    cnn = vgg16_modified()
    conv_exp = nn.Sequential(
        nn.Conv2d(512, 2048, [1, 1], 1, 0, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU()
    )
    #init conv_exp
    resnet = tv.models.resnet50(pretrained=True)
    conv_exp[0].weight.data.copy_(resnet.layer4[2].conv3.weight.data)
    conv_exp[1].weight.data.copy_(resnet.layer4[2].bn3.weight.data)

    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbIterCNN( cnn, conv_exp, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_verbiter_resnetfeatextract(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    cnn = resnet_152_features()

    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbIter_Resnet_FeatExtract( cnn, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)

def build_baseline0grid_imsitu_verb_role_joint_eval(dataset, encoder, verb_module, role_module):
    return BaseModelGrid_Imsitu_Verb_Role_Joint_Eval(encoder, verb_module, role_module)


'''def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)'''
