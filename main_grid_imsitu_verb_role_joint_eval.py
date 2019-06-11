import torch
from imsitu_encoder_alldata_imsitu import imsitu_encoder
from imsitu_loader import imsitu_loader_roleq_buatt_with_cnn
from imsitu_scorer_log import imsitu_scorer
import json
import os
import utils_imsitu
import time
import random
#from torchviz import make_dot
#from graphviz import Digraph


from dataset import Dictionary
import base_model


def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    val_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img_id, img, verb, questions, labels) in enumerate(dev_loader):
            #prit("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            t0 = time.time()
            t1 = time.time()

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                questions = torch.autograd.Variable(questions.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                questions = torch.autograd.Variable(questions)
                labels = torch.autograd.Variable(labels)

            '''print('all inputs')
            print(img)
            print('=========================================================================')
            print(verb)
            print('=========================================================================')
            print(roles)
            print('=========================================================================')
            print(labels)'''

            verb_predict, role_predict = model(img_id, img, verb, labels)
            '''loss = model.calculate_eval_loss(verb_predict, verb, role_predict, labels)
            val_loss += loss.item()'''
            top1.add_point_eval5_log_sorted(img_id, verb_predict, verb, role_predict, labels)
            top5.add_point_eval5_log_sorted(img_id, verb_predict, verb, role_predict, labels)

            del  img, verb_predict, verb, labels
            #break

    #return top1, top5, val_loss/mx

    return top1, top5, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    #parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--pretrained_buatt_model', type=str, default='', help='pretrained verb module')
    parser.add_argument('--train_role', action='store_true', help='cnn fix, verb fix, role train from the scratch')
    parser.add_argument('--use_pretrained_buatt', action='store_true', help='cnn fix, verb finetune, role train from the scratch')
    parser.add_argument('--finetune_cnn', action='store_true', help='cnn finetune, verb finetune, role train from the scratch')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--frcnn_feat_dir', type=str, help='Location of output from detectron')
    parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
    parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
    parser.add_argument('--model_saving_name', type=str, help='save name of the outpul model')
    parser.add_argument('--verb_module', type=str, default='', help='pretrained verb module')
    parser.add_argument('--role_module', type=str, default='', help='pretrained role module')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0grid_imsitu_verb_role_joint_eval')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--num_iter_verb', type=int, default=1)
    parser.add_argument('--num_iter_role', type=int, default=1)

    #verb special args

    #todo: train role module separately with gt verbs

    args = parser.parse_args()

    clip_norm = 0.25
    n_epoch = args.epochs
    batch_size = args.batch_size
    n_worker = 3

    #dataset_folder = 'imSitu'
    #imgset_folder = 'resized_256'
    dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    print('model spec :iter verb1')

    train_set = json.load(open(dataset_folder + '/' + args.train_file))
    imsitu_roleq = json.load(open("data/imsitu_questions_prev.json"))

    dict_path = 'data/dictionary_imsitu_final.pkl'
    dictionary = Dictionary.load_from_file(dict_path)
    w_emb_path = 'data/glove6b_init_imsitu_final_300d.npy'
    encoder = imsitu_encoder(train_set, imsitu_roleq, dictionary)

    train_set = imsitu_loader_roleq_buatt_with_cnn(imgset_folder, train_set, encoder, dictionary, 'train', encoder.train_transform)

    #get role_model
    print('building role model')
    role_constructor = 'build_%s' % 'baseline0grid_imsitu_roleiter_with_cnn'
    role_model = getattr(base_model, role_constructor)(train_set, args.num_hid, encoder.get_num_labels(), encoder, args.num_iter_role)

    print('building verb model')
    verb_constructor = 'build_%s' % 'baseline0grid_imsitu_roleverb_general_with_cnn'
    verb_model = getattr(base_model, verb_constructor)(train_set, args.num_hid, encoder.get_num_verbs(), encoder, role_model, args.num_iter_verb)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_set, encoder, verb_model, role_model)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader_roleq_buatt_with_cnn(imgset_folder, dev_set, encoder, dictionary, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader_roleq_buatt_with_cnn(imgset_folder, test_set, encoder, dictionary, 'test', encoder.dev_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.manual_seed(args.seed)
    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.use_pretrained_buatt:
        print('Use pretrained from: {}'.format(args.pretrained_buatt_model))
        if len(args.pretrained_buatt_model) == 0:
            raise Exception('[pretrained buatt module] not specified')
        #model_data = torch.load(args.pretrained_ban_model, map_location='cpu')
        #model.load_state_dict(model_data.get('model_state', model_data))

        utils_imsitu.load_net_ban(args.pretrained_buatt_model, [model], ['module'], ['convnet', 'w_emb', 'role_module', 'classifier'])
        model_name = 'pre_trained_buatt'

        utils_imsitu.set_trainable(model, True)
        utils_imsitu.load_net(args.role_module, [model.role_module])
        utils_imsitu.set_trainable(model.role_module, False)
        #flt img param

        opts = [{'params': model.convnet.parameters(), 'lr': 5e-5},
                {'params': model.classifier.parameters()},
                {'params': model.w_emb.parameters()},
                {'params': model.q_emb.parameters(), 'lr': 1e-4},
                {'params': model.v_att.parameters(), 'lr': 5e-5},
                {'params': model.q_net.parameters(), 'lr': 5e-5},
                {'params': model.v_net.parameters(), 'lr': 5e-5}
                ]

        optimizer = torch.optim.Adamax(opts, lr=1e-3)
        #optimizer = torch.optim.SGD(opts, lr=0.001, momentum=0.9)

    elif args.resume_training:
        print('Resume training ')
        args.train_all = True
        '''if len(args.resume_model) == 0:
            raise Exception('[pretrained verb module] not specified')'''
        utils_imsitu.load_net(args.verb_module, [model.verb_module])
        utils_imsitu.load_net(args.role_module, [model.role_module])
        optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        print('Training from the scratch.')
        model_name = 'train_full'

        utils_imsitu.set_trainable(model, True)
        utils_imsitu.load_net(args.role_module, [model.role_module])
        utils_imsitu.set_trainable(model.role_module, False)
        opts = [{'params': model.classifier.parameters()},
                {'params': model.v_att.parameters()},
                {'params': model.q_net.parameters()},
                {'params': model.v_net.parameters()},
                {'params': model.w_emb.parameters()},
                {'params': model.q_emb.parameters()}
                ]
        optimizer = torch.optim.Adamax(opts, lr=1e-3)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #utils_imsitu.set_trainable(model, True)
    #optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #gradient clipping, grad check
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if args.evaluate:
        top1, top5, val_loss = eval(model, dev_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils_imsitu.format_dict(top1_avg,'{:.2f}', '1-'),
                                                   utils_imsitu.format_dict(top5_avg, '{:.2f}', '5-')))

        #write results to csv file
        '''role_dict = top1.role_dict
        fail_val_all = top1.value_all_dict
        pass_val_dict = top1.vall_all_correct'''

        '''with open('role_pred_data.json', 'w') as fp:
            json.dump(role_dict, fp, indent=4)

        with open('fail_val_all.json', 'w') as fp:
            json.dump(fail_val_all, fp, indent=4)

        with open('pass_val_all.json', 'w') as fp:
            json.dump(pass_val_dict, fp, indent=4)'''

        '''q_dict = encoder.created_verbq_dict
        with open('createdq_roleverbgeneral_gttrain.json', 'w') as fp:
            json.dump(q_dict, fp, indent=4)

        agentplace_dict = encoder.pred_agent_place_dict
        with open('predagentplace_roleverbgeneral_gttrain.json', 'w') as fp:
            json.dump(agentplace_dict, fp, indent=4)

        topk_agentplace_dict = encoder.topk_agentplace_details
        with open('pred_top10_agentplace_roleverbgeneral_gttrain.json', 'w') as fp:
            json.dump(topk_agentplace_dict, fp, indent=4)

        all = top1.all_res

        with open('all_pred_roleverb_general_gttrain.json', 'w') as fp:
            json.dump(all, fp, indent=4)

        print('Writing predictions to file completed !')'''

    elif args.test:
        top1, top5, val_loss = eval(model, test_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                    utils_imsitu.format_dict(top1_avg,'{:.2f}', '1-'),
                                                    utils_imsitu.format_dict(top5_avg, '{:.2f}', '5-')))


    else:

        print('Model training started!')
        train(model, train_loader, dev_loader, None, optimizer, scheduler, n_epoch, args.output_dir, encoder, args.gpuid, clip_norm, None, model_name, args.model_saving_name,
              args)






if __name__ == "__main__":
    main()












