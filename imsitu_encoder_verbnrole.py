import torch
import random
from collections import OrderedDict
import csv
import nltk
import torchvision as tv
import json
import utils

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set, role_questions, roleq_dict, verbq_dict):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = []
        self.agent_label_list = []
        self.place_label_list = []
        label_frequency = {}
        self.max_role_count = 0
        self.question_words = {}
        self.max_q_word_count = 0
        self.vrole_question = {}
        self.roleq_dict = roleq_dict
        self.verbq_dict = verbq_dict
        self.verb2word_map = json.load(open('data/verb2word_mapping.json'))
        self.obj_label2eng = json.load(open('data/allwords4verbq1.json'))
        self.q_templates = json.load(open('data/role_detailed_templates.json'))
        self.all_labels = json.load(open('data/all_label_mapping.json'))
        self.created_verbq_dict = {}
        self.pred_agent_place_dict = {}
        self.verb_details = {}
        self.topk_agentplace_details = {}

        self.agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                            'seller', 'boaters', 'blocker', 'farmer']

        # imag preprocess
        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomRotation(10),
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])
        ##############################

        for verb, values in role_questions.items():
            roles = values['roles']
            verb_name = self.verb2word_map[verb][0]

            has_agent = False
            agent_role = None
            if 'agent' in roles.keys():
                agent_role = 'agent'
                has_agent = True
            else:
                for role1 in roles.keys():
                    if role1 in self.agent_roles[1:]:
                        agent_role = role1
                        has_agent = True
                        break

            for role, info in roles.items():
                #question = info['question']

                if has_agent and role == agent_role:
                    question = 'who is the ' + role + ' ' + verb_name
                elif role == 'place':
                    question = 'where is the ' + role + ' ' + verb_name
                else:
                    question = 'what is the ' + role + ' ' + verb_name

                self.vrole_question[verb+'_'+role] = question
                words = question.split()
                if len(words) > self.max_q_word_count:
                    self.max_q_word_count = len(words)
                #print('q words :', words)

                for word in words:
                    if word not in self.question_words:
                        self.question_words[word] = len(self.question_words)

        for img_id in train_set:
            img = train_set[img_id]
            current_verb = img['verb']
            if current_verb not in self.verb_list:
                self.verb_list.append(current_verb)
                self.verb2_role_dict[current_verb] = []
            roles = img['frames'][0].keys()
            has_agent = False
            has_place = False
            agent_role = None
            if 'place' in roles:
                has_place = True
            if 'agent' in roles:
                agent_role = 'agent'
                has_agent = True
            else:
                for role1 in roles:
                    if role1 in self.agent_roles[1:]:
                        agent_role = role1
                        has_agent = True
                        break

            self.verb_details[current_verb] = {'has_agent' : has_agent, 'has_place':has_place}

            for frame in img['frames']:
                for role,label in frame.items():
                    if role not in self.role_list:
                        self.role_list.append(role)
                    if role not in self.verb2_role_dict[current_verb]:
                        self.verb2_role_dict[current_verb].append(role)
                    if len(self.verb2_role_dict[current_verb]) > self.max_role_count:
                        self.max_role_count = len(self.verb2_role_dict[current_verb])
                    if label not in self.label_list:
                        '''if label not in label_frequency:
                            label_frequency[label] = 1
                        else:
                            label_frequency[label] += 1
                        #only labels occur at least 20 times are considered
                        if label_frequency[label] == 20:
                            self.label_list.append(label)'''
                        self.label_list.append(label)
                    if label not in self.agent_label_list:
                        if has_agent and role == agent_role:
                            self.agent_label_list.append(label)

                    if label not in self.place_label_list:
                        if has_place and role == 'place':
                            self.place_label_list.append(label)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t max role count:', self.max_role_count,
              '\n\t max q word count:', self.max_q_word_count)

        print('q words count :', len(self.question_words))


        verb2role_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_verb = []
            for role in current_role_list:
                role_id = self.role_list.index(role)
                role_verb.append(role_id)

            padding_count = self.max_role_count - len(current_role_list)

            for i in range(padding_count):
                role_verb.append(len(self.role_list))

            verb2role_list.append(torch.tensor(role_verb))

        self.verb2role_list = torch.stack(verb2role_list)
        self.verb2role_encoding = self.get_verb2role_encoding()
        self.verb2role_oh_encoding = self.get_verb2role_oh_encoding()
        '''print('verb to role list :', self.verb2role_list.size())

        print('unit test verb and roles: \n')
        verb_test = [4,57,367]
        for verb_id in verb_test:
            print('verb :', self.verb_list[verb_id])

            role_list = self.verb2role_list[verb_id]

            for role in role_list:
                if role != len(self.role_list):
                    print('role : ', self.role_list[role])'''

    def encode(self, item):
        verb = self.verb_list.index(item['verb'])
        roles = self.get_role_ids(verb)
        role_qs, q_len = self.get_role_questions(item['verb'])
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, roles,role_qs, q_len, labels

    def encode_ban(self, item):
        verb = self.verb_list.index(item['verb'])
        role_nl_qs = self.get_role_nl_questions(item['verb'])
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, role_nl_qs, labels

    def encode_with_rolenames(self, item):
        verb = self.verb_list.index(item['verb'])
        role_names = self.get_role_names(item['verb'])
        labels = self.get_label_ids(item['frames'])

        return verb, role_names, labels

    def encode_verb(self, item):
        verb = self.verb_list.index(item['verb'])
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, labels

    def encode_verb_only(self, item):
        verb = self.verb_list.index(item['verb'])

        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, labels

    def get_verb2role_encoding(self):
        verb2role_embedding_list = []

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []

            for role in current_role_list:
                role_embedding_verb.append(1)


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(0)

            verb2role_embedding_list.append(torch.tensor(role_embedding_verb))

        return verb2role_embedding_list

    def get_verb2role_oh_encoding(self):
        verb2role_oh_embedding_list = []

        role_oh = torch.eye(len(self.role_list)+1)

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []

            for role in current_role_list:
                role_embedding_verb.append(role_oh[self.role_list.index(role)])


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(role_oh[-1])

            verb2role_oh_embedding_list.append(torch.stack(role_embedding_verb, 0))

        return verb2role_oh_embedding_list

    def get_role_names(self, verb):
        current_role_list = self.verb2_role_dict[verb]

        role_verb = []
        for role in current_role_list:
            role_verb.append(self.role_corrected_dict[role])

        return role_verb

    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_max_role_count(self):
        return self.max_role_count

    def get_num_verbs(self):
        return len(self.verb_list)

    def get_num_roles(self):
        return len(self.role_list)

    def get_num_labels(self):
        return len(self.label_list)

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def get_role_ids_batch(self, verbs):
        role_batch_list = []
        q_len = []

        for verb_id in verbs:
            role_ids = self.get_role_ids(verb_id)
            role_batch_list.append(role_ids)

        return torch.stack(role_batch_list,0)

    def get_role_questions_batch(self, verbs):
        role_batch_list = []
        q_len_batch = []

        for verb_id in verbs:
            rquestion_tokens = []
            q_len = []
            verb = self.verb_list[verb_id]
            current_role_list = self.verb2_role_dict[verb]

            for role in current_role_list:
                question = self.vrole_question[verb+'_'+role]
                #print('question :', question)
                q_tokens = []
                words = nltk.word_tokenize(question)
                words = words[:-1]
                for word in words:
                    q_tokens.append(self.question_words[word])
                padding_words = self.max_q_word_count - len(q_tokens)

                for w in range(padding_words):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))
                q_len.append(len(words))

            role_padding_count = self.max_role_count - len(current_role_list)

            #todo : how to handle below sequence making for non roles properly?
            for i in range(role_padding_count):
                q_tokens = []
                for k in range(0,self.max_q_word_count):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))
                q_len.append(0)
            role_batch_list.append(torch.stack(rquestion_tokens,0))
            q_len_batch.append(torch.tensor(q_len))

        return torch.stack(role_batch_list,0), torch.stack(q_len_batch,0)

    def get_role_ids(self, verb_id):

        return self.verb2role_list[verb_id]

    def get_role_questions(self, verb):
        rquestion_tokens = []
        q_len = []
        current_role_list = self.verb2_role_dict[verb]

        for role in current_role_list:
            question = self.vrole_question[verb+'_'+role]
            #print('question :', question)
            q_tokens = []
            words = nltk.word_tokenize(question)
            words = words[:-1]
            for word in words:
                q_tokens.append(self.question_words[word])
            padding_words = self.max_q_word_count - len(q_tokens)

            for w in range(padding_words):
                q_tokens.append(len(self.question_words))

            rquestion_tokens.append(torch.tensor(q_tokens))
            q_len.append(len(words))

        role_padding_count = self.max_role_count - len(current_role_list)

        #todo : how to handle below sequence making for non roles properly?
        for i in range(role_padding_count):
            q_tokens = []
            for k in range(0,self.max_q_word_count):
                q_tokens.append(len(self.question_words))

            rquestion_tokens.append(torch.tensor(q_tokens))
            q_len.append(0)

        return torch.stack(rquestion_tokens,0), torch.tensor(q_len)

    def get_role_nl_questions(self, verb):
        questions = []
        current_role_list = self.verb2_role_dict[verb]

        for role in current_role_list:
            question = self.vrole_question[verb+'_'+role]
            #print('question :', question)
            questions.append(question)

        return questions

    def get_label_ids(self, frames):
        all_frame_id_list = []
        for frame in frames:
            label_id_list = []
            for role,label in frame.items():
                #use UNK when unseen labels come
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('#UNK#')

                label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(self.get_num_labels())

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels

    def get_adj_matrix(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_adj_matrix_noself(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx1 in range(0,role_count):
                adj[idx1][idx1] = 0
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def getadj(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            '''encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose'''
            adj = torch.zeros(6, 6)
            for idx in range(0,6):
                adj[idx][idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_mask(self, verb_ids, org_tensor):
        org = org_tensor.clone()
        org_reshaped = org.view(len(verb_ids), self.max_role_count, -1, org.size(2))
        for i in range(0, len(verb_ids)):
            role_encoding = self.verb2role_encoding[verb_ids[i]]
            for j in range(0, len(role_encoding)):
                #print('i:j', i,j)
                if role_encoding[j] == 0:
                    org_reshaped[i][j:] = 0
                    break
        return org_reshaped.view(org_tensor.size())

    def get_extended_encoding(self, verb_ids, dim):
        encoding_list = []
        for id in verb_ids:
            encoding = self.verb2role_encoding[id]

            encoding = torch.unsqueeze(torch.tensor(encoding),1)
            #print('encoding unsqe :', encoding.size())
            encoding = encoding.repeat(1,dim)
            #encoding = torch.squeeze(encoding)
            #print('extended :', encoding.size(), encoding)
            encoding_list.append(encoding)

        return torch.stack(encoding_list).type(torch.FloatTensor)

    def get_adj_matrix_noself_expanded(self, verb_ids, dim):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx1 in range(0,role_count):
                adj[idx1][idx1] = 0
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj = adj.unsqueeze(-1)
            adj = adj.expand(adj.size(0), adj.size(1), dim)
            adj_matrix_list.append(adj)


        return torch.stack(adj_matrix_list, 0).type(torch.FloatTensor)

    def get_adj_matrix_expanded(self, verb_ids, dim):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose

            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj = adj.unsqueeze(-1)
            adj = adj.expand(adj.size(0), adj.size(1), dim)
            adj_matrix_list.append(adj)


        return torch.stack(adj_matrix_list, 0).type(torch.FloatTensor)

    def get_role_encoding(self, verb_ids):
        verb_role_oh_list = []

        for id in verb_ids:
            verb_role_oh_list.append(self.verb2role_oh_encoding[id])

        return torch.stack(verb_role_oh_list).type(torch.FloatTensor)

    def get_generalq(self):
        general_q = "what is the action happening"

        tokens = self.verbq_dict.tokenize(general_q, False)

        return torch.tensor(tokens)

    def get_role_q_by_verb(self, verbs):
        role_qs_all = []
        max_len = 0
        for verb in verbs:
            role_nl_qs = self.get_role_nl_questions(self.verb_list[verb])

            if len(role_nl_qs) < self.max_role_count:
                padding = self.max_role_count - len(role_nl_qs)

                padded = ["" for x in range(padding)]
                role_nl_qs.extend(padded)

            role_qs_all.extend(role_nl_qs)

        for q in role_qs_all:
            length = len(q.split())
            if length > max_len:
                max_len = length

        rquestion_tokens = []
        for entry in role_qs_all:
            if len(entry) > 0:
                tokens = self.roleq_dict.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.roleq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.roleq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_agent_nl_question(self, verb):
        current_role_list = self.verb2_role_dict[verb]

        has_agent = False
        agent_role = None
        if 'agent' in current_role_list:
            agent_role = 'agent'
            has_agent = True
        else:
            for role1 in current_role_list:
                if role1 in self.agent_roles[1:]:
                    agent_role = role1
                    has_agent = True
                    break

        if has_agent:
            question = self.vrole_question[verb+'_'+agent_role]

        else:
            question = "who is the agent"

        return question

    def get_place_nl_question(self, verb):
        current_role_list = self.verb2_role_dict[verb]

        has_place = False
        place_role = None
        if 'place' in current_role_list:
            place_role = 'place'
            has_place = True


        if has_place:
            question = self.vrole_question[verb+'_'+place_role]

        else:
            question = "where is the place"

        return question

    def get_agent_place_roleqs(self, batch_size, verbs):
        role_qs_all = []
        max_len = 0
        for i in range(batch_size):
            if verbs is not None:
                verb = verbs[i]
                agent_q = self.get_agent_nl_question(self.verb_list[verb])
                place_q = self.get_place_nl_question(self.verb_list[verb])

            else:
                agent_q = "who is the agent"
                place_q = "where is the place"

            role_nl_qs = [agent_q, place_q]

            role_qs_all.extend(role_nl_qs)

        for q in role_qs_all:
            length = len(q.split())
            if length > max_len:
                max_len = length

        rquestion_tokens = []
        for entry in role_qs_all:
            if len(entry) > 0:
                tokens = self.roleq_dict.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.roleq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.roleq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_agent_roleqs(self, batch_size, verbs):
        role_qs_all = []
        max_len = 0
        for i in range(batch_size):
            if verbs is not None:
                verb = verbs[i]
                agent_q = self.get_agent_nl_question(self.verb_list[verb])

            else:
                agent_q = "who is the agent"

            role_nl_qs = [agent_q]

            role_qs_all.extend(role_nl_qs)

        for q in role_qs_all:
            length = len(q.split())
            if length > max_len:
                max_len = length

        rquestion_tokens = []
        for entry in role_qs_all:
            if len(entry) > 0:
                tokens = self.roleq_dict.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.roleq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.roleq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_place_roleqs(self, batch_size, verbs):
        role_qs_all = []
        max_len = 0
        for i in range(batch_size):
            if verbs is not None:
                verb = verbs[i]
                place_q = self.get_place_nl_question(self.verb_list[verb])

            else:
                place_q = "where is the place"

            role_nl_qs = [place_q]

            role_qs_all.extend(role_nl_qs)

        for q in role_qs_all:
            length = len(q.split())
            if length > max_len:
                max_len = length

        rquestion_tokens = []
        for entry in role_qs_all:
            if len(entry) > 0:
                tokens = self.roleq_dict.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.roleq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.roleq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_with_agentplace(self, img_id, batch_size, agent_place_ids):
        batch_size = batch_size
        all_qs = []
        max_len = 0

        for i in range(batch_size):
            im_id = img_id[i]
            current_labels = agent_place_ids[i]
            agent_name = self.label_list[current_labels[0]]

            place_name = self.label_list[current_labels[1]]
            #current_beam = top5[i]
            beam_agents = []
            beam_places = []
            '''for beamid in range(10):
                beam_agent = self.label_list[current_beam[0][beamid]]
                beam_place = self.label_list[current_beam[1][beamid]]

                beam_agents.append(beam_agent)
                beam_places.append(beam_place)'''
                #print('max agent place:', beamid, im_id, agent_name, place_name, beam_agent, beam_place)

            self.pred_agent_place_dict[im_id] = {'agent':agent_name, 'place':place_name}
            self.topk_agentplace_details[im_id] = {'agents':beam_agents, 'places':beam_places}

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_predtemplate_goldlabels(self, img_id, verb_ids, label_ids, agent_place_ids):
        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0

        for i in range(batch_size):
            im_id = img_id[i]
            current_labels = agent_place_ids[i]
            agent_name = self.label_list[current_labels[0]]

            place_name = self.label_list[current_labels[1]]


            ##### gold things ################

            curr_verb_id = verb_ids[i]
            current_labels = label_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]
            gold_agent_name = ''

            gold_place_name = ''
            if 'place' in current_role_list :
                plz_idx = current_role_list.index('place')
                gold_place_name = self.label_list[current_labels[plz_idx]]

            has_agent = False
            for role in current_role_list:
                if role in self.agent_roles:
                    has_agent = True
                    break

            if has_agent:
                if 'agent' in current_role_list:
                    agent_idx = current_role_list.index('agent')
                else:
                    for a_role in self.agent_roles[1:]:
                        if a_role in current_role_list:
                            agent_idx = current_role_list.index(a_role)
                            break
                gold_agent_name = self.label_list[current_labels[agent_idx]]


            ######################################

            self.pred_agent_place_dict[im_id] = {'agent':agent_name, 'place':place_name}

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[gold_agent_name]
                place_eng_name = self.obj_label2eng[gold_place_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[gold_place_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[gold_agent_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_with_agentplace_with_verb(self, img_id, batch_size, agent_place_ids, verbs):
        batch_size = batch_size
        all_qs = []
        max_len = 0

        for i in range(batch_size):
            im_id = img_id[i]
            current_labels = agent_place_ids[i]

            curr_verb_id = verbs[i]
            verb_name = self.verb_list[curr_verb_id]
            verb_infor_agent_place = self.verb_details[verb_name]

            agent_name = self.label_list[current_labels[0]]

            place_name = self.label_list[current_labels[1]]

            if not verb_infor_agent_place['has_place']:
                place_name = ''
            if not verb_infor_agent_place['has_agent']:
                agent_name = ''

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_with_agentplace_special(self, img_id, batch_size, agent_id, place_id):
        batch_size = batch_size
        all_qs = []
        max_len = 0

        for i in range(batch_size):
            im_id = img_id[i]
            agent_name = self.agent_label_list[agent_id[i]]

            place_name = self.place_label_list[place_id[i]]

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_idx(self, img_id, verb_ids, label_ids):
        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0
        agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                       'seller', 'boaters', 'blocker', 'farmer']
        for i in range(batch_size):
            im_id = img_id[i]
            curr_verb_id = verb_ids[i]
            current_labels = label_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]
            agent_name = ''

            place_name = ''
            if 'place' in current_role_list :
                plz_idx = current_role_list.index('place')
                place_name = self.label_list[current_labels[plz_idx]]

            has_agent = False
            for role in current_role_list:
                if role in agent_roles:
                    has_agent = True
                    break

            if has_agent:
                if 'agent' in current_role_list:
                    agent_idx = current_role_list.index('agent')
                else:
                    for a_role in agent_roles[1:]:
                        if a_role in current_role_list:
                            agent_idx = current_role_list.index(a_role)
                            break
                agent_name = self.label_list[current_labels[agent_idx]]

            #print('agent place', verb_name, agent_name,  place_name)

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_goldtemplate_predlabels(self, img_id, verb_ids, label_ids, agent_place_ids):
        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0
        agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                       'seller', 'boaters', 'blocker', 'farmer']
        for i in range(batch_size):
            im_id = img_id[i]
            curr_verb_id = verb_ids[i]

            current_pred_labels = agent_place_ids[i]
            agent_pred_name = self.label_list[current_pred_labels[0]]

            place_pred_name = self.label_list[current_pred_labels[1]]

            current_labels = label_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]
            agent_name = ''

            place_name = ''
            if 'place' in current_role_list :
                plz_idx = current_role_list.index('place')
                place_name = self.label_list[current_labels[plz_idx]]

            has_agent = False
            for role in current_role_list:
                if role in agent_roles:
                    has_agent = True
                    break

            if has_agent:
                if 'agent' in current_role_list:
                    agent_idx = current_role_list.index('agent')
                else:
                    for a_role in agent_roles[1:]:
                        if a_role in current_role_list:
                            agent_idx = current_role_list.index(a_role)
                            break
                agent_name = self.label_list[current_labels[agent_idx]]

            #print('agent place', verb_name, agent_name,  place_name)

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_pred_name]
                place_eng_name = self.obj_label2eng[place_pred_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_pred_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_pred_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_goldtemplate_predagent(self, img_id, verb_ids, label_ids, agent_place_ids):
        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0
        agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                       'seller', 'boaters', 'blocker', 'farmer']
        for i in range(batch_size):
            im_id = img_id[i]
            curr_verb_id = verb_ids[i]

            current_pred_labels = agent_place_ids[i]
            agent_pred_name = self.label_list[current_pred_labels[0]]

            place_pred_name = self.label_list[current_pred_labels[1]]

            current_labels = label_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]
            agent_name = ''

            place_name = ''
            if 'place' in current_role_list :
                plz_idx = current_role_list.index('place')
                place_name = self.label_list[current_labels[plz_idx]]

            has_agent = False
            for role in current_role_list:
                if role in agent_roles:
                    has_agent = True
                    break

            if has_agent:
                if 'agent' in current_role_list:
                    agent_idx = current_role_list.index('agent')
                else:
                    for a_role in agent_roles[1:]:
                        if a_role in current_role_list:
                            agent_idx = current_role_list.index(a_role)
                            break
                agent_name = self.label_list[current_labels[agent_idx]]

            #print('agent place', verb_name, agent_name,  place_name)

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_pred_name]
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_pred_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_goldtemplate_predplace(self, img_id, verb_ids, label_ids, agent_place_ids):
        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0
        agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                       'seller', 'boaters', 'blocker', 'farmer']
        for i in range(batch_size):
            im_id = img_id[i]
            curr_verb_id = verb_ids[i]

            current_pred_labels = agent_place_ids[i]
            agent_pred_name = self.label_list[current_pred_labels[0]]

            place_pred_name = self.label_list[current_pred_labels[1]]

            current_labels = label_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]
            agent_name = ''

            place_name = ''
            if 'place' in current_role_list :
                plz_idx = current_role_list.index('place')
                place_name = self.label_list[current_labels[plz_idx]]

            has_agent = False
            for role in current_role_list:
                if role in agent_roles:
                    has_agent = True
                    break

            if has_agent:
                if 'agent' in current_role_list:
                    agent_idx = current_role_list.index('agent')
                else:
                    for a_role in agent_roles[1:]:
                        if a_role in current_role_list:
                            agent_idx = current_role_list.index(a_role)
                            break
                agent_name = self.label_list[current_labels[agent_idx]]

            #print('agent place', verb_name, agent_name,  place_name)

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                place_eng_name = self.obj_label2eng[place_pred_name]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.obj_label2eng[place_pred_name]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.obj_label2eng[agent_name]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            self.created_verbq_dict[im_id] = question

            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.verbq_dict.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.verbq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.verbq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_detailed_roleq_idx(self, verb_ids, label_ids):

        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0



        for i in range(batch_size):
            curr_verb_id = verb_ids[i]
            current_labels = label_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]

            role_q_templates = self.q_templates[verb_name]['roles']
            current_verb_qs = []

            for i in range(len(current_role_list)):
                org_template = role_q_templates[current_role_list[i]]
                template = org_template
                for j in range(len(current_role_list)):
                    if i != j:
                        token = '<' + current_role_list[j].upper() + '>'
                        label = self.label_list[current_labels[j]]
                        label_name = self.all_labels[label] if label in self.all_labels else label
                        template = template.replace(token, label_name)

                length = len(template.split())
                if length > max_len:
                    max_len = length
                current_verb_qs.append(template)
            all_qs.append(current_verb_qs)

        all_new_list = []
        for q_list in all_qs:
            rquestion_tokens = []
            for entry in q_list:
                #print('tokeningzing :', entry)
                tokens = self.roleq_dict.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.roleq_dict.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))

            role_padding_count = self.max_role_count - len(rquestion_tokens)

            #todo : how to handle below sequence making for non roles properly?
            for i in range(role_padding_count):
                padding = [self.roleq_dict.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(padding))

            all_new_list.append(torch.stack(rquestion_tokens,0))

        return torch.stack(all_new_list,0)


