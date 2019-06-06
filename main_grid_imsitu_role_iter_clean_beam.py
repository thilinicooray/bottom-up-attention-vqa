import torch
from imsitu_encoder_alldata_imsitu import imsitu_encoder
from imsitu_loader import imsitu_loader_roleq_buatt
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


def train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, lr_max, model_name, model_saving_name, args,eval_frequency=4000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []
    time_all = time.time()

    '''if gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]

        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model'''
    pmodel = model

    '''if scheduler.get_lr()[0] < lr_max:
        scheduler.step()'''

    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)

    '''print('init param data check :')
    for f in model.parameters():
        if f.requires_grad:
            print(f.data.size())'''


    for epoch in range(max_epoch):

        #print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        #sizes batch_size*3*height*width, batch*504*1, batch*6*190*1, batch*3*6*lebale_count*1
        mx = len(train_loader)
        for i, (_, img, verb, questions, labels) in enumerate(train_loader):
            #print("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            t0 = time.time()
            t1 = time.time()
            total_steps += 1

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

            role_predict, loss1 = pmodel(img, questions, labels, verb)
            #verb_predict, rol1pred, role_predict = pmodel.forward_eval5(img)
            #print ("forward time = {}".format(time.time() - t1))
            t1 = time.time()
            loss = loss1

            '''g = make_dot(verb_predict, model.state_dict())
            g.view()'''

            #loss = model.calculate_loss(verb, role_predict, labels, args)
            #loss = model.calculate_eval_loss_new(verb_predict, verb, rol1pred, labels, args)
            #loss = loss_ * random.random() #try random loss
            #print ("loss time = {}".format(time.time() - t1))
            t1 = time.time()
            #print('current loss = ', loss)

            loss.backward()
            #print ("backward time = {}".format(time.time() - t1))

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)


            '''for param in filter(lambda p: p.requires_grad,model.parameters()):
                print(param.grad.data.sum())'''

            #start debugger
            #import pdb; pdb.set_trace()


            optimizer.step()
            optimizer.zero_grad()

            '''print('grad check :')
            for f in model.parameters():
                print('data is')
                print(f.data)
                print('grad is')
                print(f.grad)'''

            train_loss += loss.item()

            #top1.add_point_eval5(verb_predict, verb, role_predict, labels)
            #top5.add_point_eval5(verb_predict, verb, role_predict, labels)

            top1.add_point_noun(verb, role_predict, labels)
            top5.add_point_noun(verb, role_predict, labels)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results_nouns()
                top5_a = top5.get_average_results_nouns()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils_imsitu.format_dict(top1_a, "{:.2f}", "1-"),
                               utils_imsitu.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results_nouns()
                top5_avg = top5.get_average_results_nouns()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils_imsitu.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils_imsitu.format_dict(top5_avg, '{:.2f}', '5-')))
                #print('Dev loss :', val_loss)

                dev_score_list.append(avg_score)
                max_score = max(dev_score_list)

                if max_score == dev_score_list[-1]:
                    torch.save(model.state_dict(), model_dir + "/{}_roles_iter_ftall_beam{}.model".format( model_name, model_saving_name))
                    print ('New best model saved! {0}'.format(max_score))

                #eval on the trainset

                '''top1, top5, val_loss = eval(model, traindev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('TRAINDEV {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                                  utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                                  utils.format_dict(top5_avg, '{:.2f}', '5-')))'''

                print('current train loss', train_loss)
                train_loss = 0
                top1 = imsitu_scorer(encoder, 1, 3)
                top5 = imsitu_scorer(encoder, 5, 3)

            del role_predict, loss, img, verb, labels
            #break
        print('Epoch ', epoch, ' completed!')
        scheduler.step()
        #break

def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    val_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img_id, img, verb, questions, labels) in enumerate(dev_loader):
            #print("{}/{} batches\r".format(i+1,mx)) ,
            '''im_data = torch.squeeze(im_data,0)
            im_info = torch.squeeze(im_info,0)
            gt_boxes = torch.squeeze(gt_boxes,0)
            num_boxes = torch.squeeze(num_boxes,0)
            verb = torch.squeeze(verb,0)
            roles = torch.squeeze(roles,0)
            labels = torch.squeeze(labels,0)'''

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

            role_predict = model.forward_eval(img, questions, verb)
            '''loss = model.calculate_eval_loss(verb_predict, verb, role_predict, labels)
            val_loss += loss.item()'''
            top1.add_point_noun_top1(img_id, verb, role_predict, labels)
            top5.add_point_noun_top1(img_id, verb, role_predict, labels)

            del role_predict, img, verb, labels
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

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0grid_imsitu_roleiter_beam')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size')

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

    print('model spec :, top down att with role q ')

    train_set = json.load(open(dataset_folder + '/' + args.train_file))
    imsitu_roleq = json.load(open("data/imsitu_questions_prev.json"))


    dict_path = 'data/dictionary_vqa_imsitu_both.pkl'
    dictionary = Dictionary.load_from_file(dict_path)
    w_emb_path = 'data/glove6b_init_vqa_imsitu_both_300d.npy'
    encoder = imsitu_encoder(train_set, imsitu_roleq, dictionary)

    train_set = imsitu_loader_roleq_buatt(imgset_folder, train_set, encoder, dictionary, 'train', encoder.train_transform)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_set, args.num_hid, encoder.get_num_labels(), encoder, args.num_iter,
                                             args.beam_size)

    model.w_emb.init_embedding(w_emb_path)

    #print('MODEL :', model)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader_roleq_buatt(imgset_folder, dev_set, encoder, dictionary, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader_roleq_buatt(imgset_folder, test_set, encoder, dictionary, 'test', encoder.dev_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #torch.manual_seed(1234)
    torch.manual_seed(args.seed)
    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    if args.use_pretrained_buatt:
        print('Use pretrained from: {}'.format(args.pretrained_buatt_model))
        if len(args.pretrained_buatt_model) == 0:
            raise Exception('[pretrained buatt module] not specified')
        #model_data = torch.load(args.pretrained_ban_model, map_location='cpu')
        #model.load_state_dict(model_data.get('model_state', model_data))

        utils_imsitu.load_net_ban(args.pretrained_buatt_model, [model], ['module'], ['classifier'])
        model_name = 'pre_trained_buatt'

        utils_imsitu.set_trainable(model, True)
        #utils_imsitu.set_trainable(model.classifier, True)
        #utils_imsitu.set_trainable(model.w_emb, False)
        #utils_imsitu.set_trainable(model.q_emb, True)
        optimizer = torch.optim.Adamax([
            {'params': model.classifier.parameters()},
            {'params': model.w_emb.parameters(), 'lr': 1e-4},
            {'params': model.q_emb.parameters(), 'lr': 1e-4},
            {'params': model.v_att.parameters(), 'lr': 1e-4},
            {'params': model.q_net.parameters(), 'lr': 1e-4},
            {'params': model.v_net.parameters(), 'lr': 1e-4},
        ], lr=1e-3)

    elif args.resume_training:
        print('Resume training from: {}'.format(args.resume_model))
        args.train_all = True
        if len(args.resume_model) == 0:
            raise Exception('[pretrained module] not specified')
        utils_imsitu.load_net(args.resume_model, [model])
        optimizer_select = 0
        model_name = 'resume_all'
    else:
        print('Training from the scratch.')
        model_name = 'train_full'
        utils_imsitu.set_trainable(model, True)
        #utils_imsitu.set_trainable(model.classifier, True)
        #utils_imsitu.set_trainable(model.w_emb, True)
        #utils_imsitu.set_trainable(model.q_emb, True)
        optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)



    #utils_imsitu.set_trainable(model, True)
    #optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    #gradient clipping, grad check
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if args.evaluate:
        top1, top5, val_loss = eval(model, dev_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils_imsitu.format_dict(top1_avg,'{:.2f}', '1-'),
                                                   utils_imsitu.format_dict(top5_avg, '{:.2f}', '5-')))

        #write results to csv file
        role_dict = top1.role_dict
        fail_val_all = top1.value_all_dict
        pass_val_dict = top1.vall_all_correct

        with open('role_pred_data.json', 'w') as fp:
            json.dump(role_dict, fp, indent=4)

        with open('fail_val_all.json', 'w') as fp:
            json.dump(fail_val_all, fp, indent=4)

        with open('pass_val_all.json', 'w') as fp:
            json.dump(pass_val_dict, fp, indent=4)

        print('Writing predictions to file completed !')

    elif args.test:
        top1, top5, val_loss = eval(model, test_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

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












