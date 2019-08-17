import torch
import random
from collections import OrderedDict
import csv
import nltk
import torchvision as tv
import json
import utils
import numpy as np

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set, role_questions, dict):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = []
        label_frequency = {}
        self.max_role_count = 0
        self.question_words = {}
        self.max_q_word_count = 0
        self.vrole_question = {}
        self.dictionary = dict
        self.q_templates = json.load(open('data/role_detailed_templates_agentctx_only.json'))
        self.all_words = json.load(open('data/allnverbsall_imsitu_words_nl2glovematching.json'))
        self.labelid2nlword = json.load(open('data/all_imsitu_words_id2nl.json'))

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
            verb_name = self.all_words[verb]

            '''has_agent = False
            agent_role = None
            if 'agent' in roles.keys():
                agent_role = 'agent'
                has_agent = True
            else:
                for role1 in roles.keys():
                    if role1 in self.agent_roles[1:]:
                        agent_role = role1
                        has_agent = True
                        break'''

            for role, info in roles.items():
                #question = info['question']

                if role in self.agent_roles:
                    question = 'who is the ' + self.all_words[role] + ' ' + verb_name
                elif role == 'place':
                    question = 'where is the ' + self.all_words[role] + ' ' + verb_name
                else:
                    question = 'what is the ' + self.all_words[role] + ' ' + verb_name

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

        verb2role_withaction_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_verb_withaction = [len(self.role_list) + 1]
            for role in current_role_list:
                role_id = self.role_list.index(role)
                role_verb_withaction.append(role_id)

            padding_count = self.max_role_count - len(current_role_list)

            for i in range(padding_count):
                role_verb_withaction.append(len(self.role_list))

            verb2role_withaction_list.append(torch.tensor(role_verb_withaction))

        self.verb2role_withaction_list = torch.stack(verb2role_withaction_list)


        self.details_of_ordered = self.get_ordered_details()



        '''print('verb to role list :', self.verb2role_list.size())

        print('unit test verb and roles: \n')
        verb_test = [4,57,367]
        for verb_id in verb_test:
            print('verb :', self.verb_list[verb_id])

            role_list = self.verb2role_list[verb_id]

            for role in role_list:
                if role != len(self.role_list):
                    print('role : ', self.role_list[role])'''


    def get_ordered_details(self):

        ordered_details = {}

        verb2role_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_verb = []

            #add place in the first place
            if 'place' in  current_role_list:
                role_id = self.role_list.index('place')
                role_verb.append(role_id)

            else:
                role_verb.append(len(self.role_list))

            #add agent to next
            agent_role = None
            if 'agent' in current_role_list:
                agent_role = 'agent'
            else:
                for role1 in current_role_list:
                    if role1 in self.agent_roles[1:]:
                        agent_role = role1
                        break

            if agent_role is not None:
                role_id = self.role_list.index(agent_role)
                role_verb.append(role_id)

            else:
                role_verb.append(len(self.role_list))


            for role in current_role_list:
                role_id = self.role_list.index(role)
                if role_id not in role_verb:
                    role_verb.append(role_id)

            padding_count = self.max_role_count - len(role_verb)

            for i in range(padding_count):
                role_verb.append(len(self.role_list))

            verb2role_list.append(torch.tensor(role_verb))

        verb2role_list = torch.stack(verb2role_list)
        ordered_details['verb2role_list'] = verb2role_list


        verb2role_encoding = []

        for verb_id in range(len(self.verb_list)):
            role_ids = verb2role_list[verb_id]
            role_embedding_verb = []

            for r_id in role_ids:
                if r_id == len(self.role_list):
                    role_embedding_verb.append(0)
                else:
                    role_embedding_verb.append(1)

            verb2role_encoding.append(torch.tensor(role_embedding_verb))

        ordered_details['verb2role_encoding'] = verb2role_encoding

        return ordered_details


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
        #role_nl_qs = self.get_role_nl_questions(item['verb'])
        labels = self.get_label_ids(item['verb'], item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, labels

    def encode_roleonly_indiloss(self, item):
        verb = self.verb_list.index(item['verb'])
        role_nl_qs = self.get_role_nl_questions(item['verb'])
        labels = self.get_label_ids(item['verb'], item['frames'])
        label_scores = self.get_label_scores(item['verb'], item['frames']) #zero vector of size vocab, filled with score for correct nouns

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, role_nl_qs, labels, label_scores

    def encode_roleonly_ordered(self, item):
        verb = self.verb_list.index(item['verb'])
        labels = self.get_label_ids_ordered(item['verb'], item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, labels

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


    def get_ordered_role_ids_batch(self, verbs):
        role_batch_list = []
        q_len = []

        for verb_id in verbs:
            role_ids = self.get_role_ids_ordered(verb_id)
            print('role ids ', role_ids)
            role_batch_list.append(role_ids)

        return torch.stack(role_batch_list,0)

    def get_role_ids_with_actionrole_batch(self, verbs):
        role_batch_list = []
        q_len = []

        for verb_id in verbs:
            role_ids = self.get_role_ids_with_actionrole(verb_id)
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


    def get_role_ids_ordered(self, verb_id):

        return self.details_of_ordered['verb2role_list'][verb_id]


    def get_role_ids_with_actionrole(self, verb_id):

        return self.verb2role_withaction_list[verb_id]

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

    def get_role_nl_questions_batch(self, verb_ids):

        batch_size = verb_ids.size(0)
        all_qs = []
        max_len = 0

        for i in range(batch_size):
            curr_verb_id = verb_ids[i]
            verb_name = self.verb_list[curr_verb_id]
            current_role_list = self.verb2_role_dict[verb_name]
            questions = []

            for role in current_role_list:
                question = self.vrole_question[verb_name+'_'+role]
                length = len(question.split())
                if length > max_len:
                    max_len = length
                questions.append(question)
            all_qs.append(questions)

        all_new_list = []
        for q_list in all_qs:
            rquestion_tokens = []
            for entry in q_list:
                #print('tokeningzing :', entry)
                tokens = self.dictionary.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))

            role_padding_count = self.max_role_count - len(rquestion_tokens)

            #todo : how to handle below sequence making for non roles properly?
            for i in range(role_padding_count):
                padding = [self.dictionary.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(padding))

            all_new_list.append(torch.stack(rquestion_tokens,0))

        return torch.stack(all_new_list,0)

    def get_label_ids(self, verb, frames):
        all_frame_id_list = []
        roles = self.verb2_role_dict[verb]
        for frame in frames:
            label_id_list = []

            for role in roles:
                label = frame[role]
                #use UNK when unseen labels come
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('#UNK#')

                label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(self.label_list.index(""))

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels


    def get_label_ids_ordered(self, verb, frames):
        all_frame_id_list = []
        print('verb ', verb)
        verb_id = self.verb_list.index(verb)
        roles = self.details_of_ordered['verb2role_list'][verb_id]
        print('roles ', roles)


        for frame in frames:
            label_id_list = []

            for role_id in roles:
                if role_id == len(self.role_list):
                    label_id_list.append(self.label_list.index(""))
                else:
                    role = self.role_list.index(role_id)
                    label = frame[role]
                    #use UNK when unseen labels come
                    if label in self.label_list:
                        label_id = self.label_list.index(label)
                    else:
                        label_id = self.label_list.index('#UNK#')

                    label_id_list.append(label_id)

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)
        print('label ', labels)

        return labels

    def get_label_scores(self, verb, frames):
        all_role_list = []
        roles = self.verb2_role_dict[verb]

        for role in roles:
            label_counts = np.zeros(self.get_num_labels()+1)
            for frame in frames:
                label = frame[role]
                label_id = self.label_list.index(label)

                label_counts[label_id] = 1

                '''if label_id in label_counts:
                    label_counts[label_id] += 1
                else:
                    label_counts[label_id] = 1'''

            #current_role_scorevec = label_counts / np.sum(label_counts)
            current_role_scorevec = label_counts
            all_role_list.append(torch.from_numpy(current_role_scorevec).float())

        role_padding_count = self.max_role_count - len(roles)

        for i in range(role_padding_count):
            pad = torch.zeros(self.get_num_labels()+1)
            pad[-1] = 1.0
            all_role_list.append(pad)

        scores = torch.stack(all_role_list,0)


        return scores

    def get_label_score_values(self, counts):
        import numpy as np

        prob_answer_vec = np.zeros(self.get_num_labels()+1)

        '''current_role_scorevec = torch.zeros(self.get_num_labels())

        for k, v in counts.items():
            score = 0
            if v == 1:
                score = 0.3
            elif v == 2:
                score = 0.6
            else:
                score = 1.0

            current_role_scorevec[k] = score'''
        for k, v in counts.items():
            prob_answer_vec[k] = v


        dis = prob_answer_vec / np.sum(prob_answer_vec)
        #print(dis)

        return torch.from_numpy(dis).float()

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

    def get_verb2role_encoing_batch(self, verb_ids):
        matrix_list = []
        role_tot = 0

        for id in verb_ids:
            encoding = self.verb2role_encoding[id]
            matrix_list.append(encoding)
            role_tot += self.get_role_count(id)

        encoding_all = torch.stack(matrix_list).type(torch.FloatTensor)

        return encoding_all, role_tot



    def get_adj_matrix_noself(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(encoding.clone().detach(), 0)
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


    def get_adj_matrix_noself_ordered(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            print('adj ids :', id)
            encoding = self.details_of_ordered['verb2role_encoding'][id]
            print('encoding ', encoding, encoding.size(0))
            encoding_tensor = torch.unsqueeze(encoding.clone().detach(), 0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose

            for idx1 in range(encoding.size(0)):
                print('idx ', idx1, idx1, encoding[idx1] == 1)
                if encoding[idx1] == 1:
                    print('came')
                    adj[idx1][idx1] = 0
                    print('now ', adj)
                else:
                    adj[idx1][idx1] = 1

            print('adj ', adj)

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

    def get_detailed_roleq_idx(self, verb_ids, label_ids):

        if label_ids is None:
            #get general roles
            return self.get_role_nl_questions_batch(verb_ids)

        else:

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
                role_dict = {}
                for j in range(len(current_role_list)):
                    label = self.label_list[current_labels[j]]
                    #remember to add 'UNK' as a key to labelid2nlword and all labels must be inside the dict
                    label_name = self.all_words[self.labelid2nlword[label]]
                    role_dict[current_role_list[j].upper()] = label_name

                for i in range(len(current_role_list)):
                    org_template = role_q_templates[current_role_list[i]]
                    template = org_template.format(**role_dict)

                    #transform all according to correct word forms
                    split_temp = template.split()
                    all_tot = []

                    for word in split_temp:
                        if word == 'agentparts':
                            print('HERERRERERERRE  ', template)
                        final_word = self.all_words[word] if word in self.all_words else word
                        all_tot.append(final_word)

                    updated_template = ' '.join(all_tot)

                    #print('template :', template, updated_template)

                    length = len(updated_template.split())
                    if length > max_len:
                        max_len = length
                    current_verb_qs.append(updated_template)
                all_qs.append(current_verb_qs)

            all_new_list = []
            for q_list in all_qs:
                rquestion_tokens = []
                for entry in q_list:
                    #print('tokeningzing :', entry)
                    tokens = self.dictionary.tokenize(entry, False)
                    tokens = tokens[:max_len]
                    if len(tokens) < max_len:
                        # Note here we pad in front of the sentence
                        padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                        tokens = tokens + padding
                    utils.assert_eq(len(tokens), max_len)
                    rquestion_tokens.append(torch.tensor(tokens))

                role_padding_count = self.max_role_count - len(rquestion_tokens)

                #todo : how to handle below sequence making for non roles properly?
                for i in range(role_padding_count):
                    padding = [self.dictionary.padding_idx] * (max_len)
                    rquestion_tokens.append(torch.tensor(padding))

                all_new_list.append(torch.stack(rquestion_tokens,0))

            return torch.stack(all_new_list,0)

    def get_detailed_roleq_idx_agentplace_ctx(self, verb_ids, label_ids):

        if label_ids is None:
            #get general roles
            return self.get_role_nl_questions_batch(verb_ids)

        else:

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
                role_dict = {}
                for j in range(len(current_role_list)):
                    label = self.label_list[current_labels[j]]
                    #remember to add 'UNK' as a key to labelid2nlword and all labels must be inside the dict
                    label_name = self.all_words[self.labelid2nlword[label]]
                    role_dict[current_role_list[j].upper()] = label_name

                for i in range(len(current_role_list)):
                    org_template = role_q_templates[current_role_list[i]]
                    template = org_template.format(**role_dict)

                    #transform all according to correct word forms
                    split_temp = template.split()
                    all_tot = []

                    for word in split_temp:
                        if word == 'agentparts':
                            print('HERERRERERERRE  ', template)
                        final_word = self.all_words[word] if word in self.all_words else word
                        all_tot.append(final_word)

                    updated_template = ' '.join(all_tot)

                    #print('template :', template, updated_template)

                    length = len(updated_template.split())
                    if length > max_len:
                        max_len = length
                    current_verb_qs.append(updated_template)
                all_qs.append(current_verb_qs)

            all_new_list = []
            for q_list in all_qs:
                rquestion_tokens = []
                for entry in q_list:
                    #print('tokeningzing :', entry)
                    tokens = self.dictionary.tokenize(entry, False)
                    tokens = tokens[:max_len]
                    if len(tokens) < max_len:
                        # Note here we pad in front of the sentence
                        padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                        tokens = tokens + padding
                    utils.assert_eq(len(tokens), max_len)
                    rquestion_tokens.append(torch.tensor(tokens))

                role_padding_count = self.max_role_count - len(rquestion_tokens)

                #todo : how to handle below sequence making for non roles properly?
                for i in range(role_padding_count):
                    padding = [self.dictionary.padding_idx] * (max_len)
                    rquestion_tokens.append(torch.tensor(padding))

                all_new_list.append(torch.stack(rquestion_tokens,0))

            return torch.stack(all_new_list,0)

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
                tokens = self.dictionary.tokenize(entry, False)
                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.dictionary.padding_idx] * (max_len)
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

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'


            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.dictionary.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.dictionary.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

    def get_verbq_with_agentplace_with_grounded_info(self, img_id, batch_size, agent_place_ids, role_rep):
        batch_size = batch_size
        all_qs = []

        max_len = 0
        agent_place_idx = []

        for i in range(batch_size):
            im_id = img_id[i]
            current_labels = agent_place_ids[i]
            agent_name = self.label_list[current_labels[0]]

            place_name = self.label_list[current_labels[1]]
            cur_agent_place_dict = {}

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
                cur_agent_place_dict['agent']= [3, 2+len(agent_eng_name.split())]
                cur_agent_place_dict['place'] = [cur_agent_place_dict ['agent'][1]+4, len(question.split()) - 1]

            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the action happening at the ' + place_eng_name
                cur_agent_place_dict['agent']= [-1, -1]
                cur_agent_place_dict['place'] = [7, len(question.split()) - 1]
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                question = 'what is the '+ agent_eng_name + ' doing'
                cur_agent_place_dict['agent']= [3, 2+len(agent_eng_name.split())]
                cur_agent_place_dict['place'] = [-1, -1]
            else:
                question = 'what is the action happening'
                cur_agent_place_dict = {'agent':[-1,-1], 'place':[-1,-1]}


            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
            agent_place_idx.append(cur_agent_place_dict)

        rquestion_tokens = []
        grounded_info = []

        for j in range(len(all_qs)):
            entry = all_qs[j]

            if len(entry) > 0:
                tokens = self.dictionary.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.dictionary.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

            current_agent_place_idx = agent_place_idx[j]
            agent_rep = role_rep[j][0]
            place_rep = role_rep[j][1]

            init_info = torch.zeros(max_len, agent_rep.size(-1))
            if current_agent_place_idx['agent'][0] != -1:
                for ag_idx in current_agent_place_idx['agent']:
                    init_info[ag_idx] = agent_rep

            if current_agent_place_idx['place'][0] != -1:
                for plz_idx in current_agent_place_idx['place']:
                    init_info[plz_idx] = place_rep

            grounded_info.append(init_info)

        #print(rquestion_tokens[0], agent_place_idx[0], grounded_info[0][agent_place_idx[0]['agent'][0]],grounded_info[0][agent_place_idx[0]['agent'][1]])

        return torch.stack(rquestion_tokens,0), torch.stack(grounded_info,0)

    def get_verbq_with_agentplace_eval(self, img_id, batch_size, agent_place_ids, label_logits_10, label_id_10):
        batch_size = batch_size
        all_qs = []
        max_len = 0
        agent_names = {}
        place_names = {}
        agent_10 = {}
        place_10 = {}


        for i in range(batch_size):
            agent_eng_name = "NONE"
            place_eng_name = "NONE"
            im_id = img_id[i]
            current_labels = agent_place_ids[i]
            agent_name = self.label_list[current_labels[0]]

            place_name = self.label_list[current_labels[1]]

            current_label_logits_10 = label_logits_10[i]
            current_label_id_10 = label_id_10[i]

            if len(agent_name) > 0 and len(place_name) > 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'

            agent_names[im_id] = agent_eng_name
            place_names[im_id] = place_eng_name


            if agent_eng_name != "NONE":
                agents_10 = []
                agent_id_list = current_label_id_10[0]
                agent_logit_list = current_label_logits_10[0]
                agent_10_logits = []

                for j in range(len(agent_id_list)):
                    ag = agent_id_list[j]
                    agents_10.append(self.all_words[self.labelid2nlword[self.label_list[ag]]])
                    agent_10_logits.append(agent_logit_list[j].item())

                agent_10[im_id] = {'names':agents_10, 'logits':agent_10_logits}

            else:
                agent_10[im_id] = {'names':[], 'logits':[]}

            if place_eng_name != "NONE":
                places_10 = []
                place_id_list = current_label_id_10[1]
                place_logit_list = current_label_logits_10[1]
                place_10_logits = []

                for i in range(len(place_id_list)):
                    pl = place_id_list[i]
                    places_10.append(self.all_words[self.labelid2nlword[self.label_list[pl]]])
                    place_10_logits.append(place_logit_list[i].item())

                place_10[im_id] = {'names':places_10, 'logits':place_10_logits}

            else:
                place_10[im_id] = {'names':[], 'logits':[]}


            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.dictionary.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.dictionary.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0), agent_names, place_names, agent_10, place_10

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
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the ' + agent_eng_name + ' doing at the ' + place_eng_name
            elif len(place_name) > 0 and len(agent_name) == 0:
                place_eng_name = self.all_words[self.labelid2nlword[place_name]]
                question = 'what is the action happening at the ' + place_eng_name
            elif len(agent_name) > 0 and len(place_name) == 0:
                agent_eng_name = self.all_words[self.labelid2nlword[agent_name]]
                question = 'what is the '+ agent_eng_name + ' doing'
            else:
                question = 'what is the action happening'


            length = len(question.split())
            if length > max_len:
                max_len = length
            all_qs.append(question)
        rquestion_tokens = []
        for entry in all_qs:
            if len(entry) > 0:
                tokens = self.dictionary.tokenize(entry, False)
                #print('question', entry, tokens)

                tokens = tokens[:max_len]
                if len(tokens) < max_len:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_len - len(tokens))
                    tokens = tokens + padding
                utils.assert_eq(len(tokens), max_len)
                rquestion_tokens.append(torch.tensor(tokens))
            else:
                tokens = [self.dictionary.padding_idx] * (max_len)
                rquestion_tokens.append(torch.tensor(tokens))

        return torch.stack(rquestion_tokens,0)

