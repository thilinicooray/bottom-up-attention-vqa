import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import torchvision as tv
import utils_imsitu
import numpy as np

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
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder):
        super(BaseModelGrid_Imsitu_Agent, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder

    def forward(self, v, q, labels, verb):
        """Forward

        v: [batch, org img grid]
        q: [batch_size, seq_length]

        return: logits, not probs
        """

        img = v
        q = q.squeeze()
        print('inside model :', img.size(), q.size())

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        loss = None
        if self.training():
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
                loss += utils_imsitu.cross_entropy_loss(role_label_pred[i], gt_labels[i,index] ,len(self.encoder.agent_label_list))

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

            verb_q_idx = self.encoder.get_detailed_roleq_idx(gt_verb, label_idx)

            if torch.cuda.is_available():
                q = verb_q_idx.to(torch.device('cuda'))

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
        img_features = self.conv_exp(self.conv(image))
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        v = img_features.view(batch_size, n_channel, -1)
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
    return BaseModelGrid_Imsitu_Agent( w_emb, q_emb, v_att, q_net, v_net, classifier, encoder)

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

def build_baseline0grid_imsitu4verb(dataset, num_hid, num_ans_classes, encoder, num_iter):
    print('words count :', encoder.roleq_dict.ntoken)
    cnn = resnet_152_features()
    conv_exp = nn.Sequential(
        nn.Conv2d(512, 2048, [1, 1], 1, 0, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU()
    )
    w_emb = WordEmbedding(encoder.roleq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    return BaseModelGrid_Imsitu_RoleIter( cnn, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, num_iter)

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

def build_baseline0grid_imsitu_verbiterCNN(dataset, num_hid, num_ans_classes, encoder, role_module, num_iter):
    print('words count verbiter:', encoder.verbq_dict.ntoken)
    cnn = vgg16_modified()
    conv_exp = nn.Sequential(
        nn.Conv2d(512, 2048, [1, 1], 1, 0, bias=False),
        nn.BatchNorm2d(2048),
        nn.ReLU()
    )
    w_emb = WordEmbedding(encoder.verbq_dict.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(2048, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([2048, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_classes, 0.5)
    role_module = role_module
    return BaseModelGrid_Imsitu_VerbIterCNN( cnn, conv_exp, w_emb, q_emb, v_att, q_net, v_net, classifier, encoder, role_module, num_iter)


'''def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)'''
