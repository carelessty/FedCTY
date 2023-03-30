from libs import *
from data import *
from models import *
from methods import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 5
alpha_coef = 1e-2
learning_rate = 0.1
print_per = epoch // 2

com_amount = 100
save_period = 50
weight_decay = 1e-3
batch_size = 50
#act_prob = 1
act_prob = 0.15
lr_decay_per_round = 0.998

def train(args):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)

    

    # Prepare dataset
    # Prepare dataloader
    train_loader, test_loader = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare model
    model_func = lambda: client_model(args.model_name)
    init_model = model_func()
    torch.manual_seed(37)

    if not os.path.exists('%sModel/%s_init_mdl.pt' % (args.data_path,  args.model_name)):
        if not os.path.exists('%sModel/%s/' % (args.data_path, args.model_name)):
            print("Create a new directory")
            os.mkdir('%sModel/%s/' % (args.data_path, args.model_name))
        torch.save(init_model.state_dict(), '%sModel/%s_init_mdl.pt' %
                   (args.data_path, args.model_name))
    else:
        # Load model
        init_model.load_state_dict(torch.load(
            '%sModel/%s_init_mdl.pt' % (args.data_path, args.model_name)))

    n_clnt = args.num_local_clients

    # 获取每个客户端的数据和标签
    clnt_x = []
    clnt_y = []
    for dl_train in train_loader:
        x, y = zip(*[batch for batch in dl_train])
        clnt_x.append(torch.cat(x, dim=0))
        clnt_y.append(torch.cat(y, dim=0))


    # 获取所有测试数据和标签
    test_x = []
    test_y = []
    for dl_test in test_loader:
        x, y = zip(*[batch for batch in dl_test])
        test_x.append(torch.cat(x, dim=0))
        test_y.append(torch.cat(y, dim=0))

    test_x = torch.cat(test_x, dim=0)
    test_y = torch.cat(test_y, dim=0)

        

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not os.path.exists('%sModel/%s' % (args.data_path, args.dataset))):
        os.mkdir('%sModel/%s' % (args.data_path, args.dataset))
    save_period = 100
    n_data_per_client = np.concatenate(clnt_x, axis=0).shape[0] / n_clnt
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
    n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)



    n_save_instances = int(args.max_communication_rounds / save_period)
    avg_ins_mdls = list(range(n_save_instances))
    avg_all_mdls = list(range(n_save_instances))
    avg_cld_mdls = list(range(n_save_instances))

    trn_sel_clt_perf = np.zeros((args.max_communication_rounds, 2))
    tst_sel_clt_perf = np.zeros((args.max_communication_rounds, 2))

    trn_all_clt_perf = np.zeros((args.max_communication_rounds, 2))
    tst_all_clt_perf = np.zeros((args.max_communication_rounds, 2))

    trn_cur_cld_perf = np.zeros((args.max_communication_rounds, 2))
    tst_cur_cld_perf = np.zeros((args.max_communication_rounds, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype(
        'float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    ###
    state_gadient_diffs = np.zeros(
        (n_clnt+1, n_par)).astype('float32')  # including cloud state

    # wandb.init(project="FedDC", name='temp')
    np.random.seed(0)
    wandb.init(project="FedDC_test2", name=str(np.random.randint(100)))
    for i in range(args.max_communication_rounds):
                if os.path.exists('%sModel/%s/ins_avg_%dcom.pt'
                                % (args.data_path, args.dataset, i+1)):
                    saved_itr = i

                    ####
                    fed_ins = model_func()
                    fed_ins.load_state_dict(torch.load(
                        '%sModel/%s/%s/ins_avg_%dcom.pt' % (args.data_path, args.dataset, 'temp', i+1)))
                    fed_ins.eval()
                    fed_ins = fed_ins.to(device)

                    for params in fed_ins.parameters():
                        params.requires_grad = False

                    avg_ins_mdls[saved_itr//save_period] = fed_ins

                    ####
                    fed_all = model_func()
                    fed_all.load_state_dict(torch.load(
                        '%sModel/%s/%s/all_avg_%dcom.pt' % (args.data_path, args.dataset, 'temp', i+1)))
                    fed_all.eval()
                    fed_all = fed_all.to(device)

                    # Freeze model
                    for params in fed_all.parameters():
                        params.requires_grad = False

                    avg_all_mdls[saved_itr//save_period] = fed_all

                    ####
                    fed_cld = model_func()
                    fed_cld.load_state_dict(torch.load(
                        '%sModel/%s/%s/cld_avg_%dcom.pt' % (args.data_path, args.dataset, 'temp', i+1)))
                    fed_cld.eval()
                    fed_cld = fed_cld.to(device)

                    # Freeze model
                    for params in fed_cld.parameters():
                        params.requires_grad = False

                    avg_cld_mdls[saved_itr//save_period] = fed_cld

                    if os.path.exists('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (args.data_path, args.dataset, 'temp', (i+1))):

                        trn_sel_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (
                            args.data_path, args.dataset, 'temp', (i+1)))
                        tst_sel_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' % (
                            args.data_path, args.dataset, 'temp', (i+1)))
                        trn_all_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' % (
                            args.data_path, args.dataset, 'temp', (i+1)))
                        tst_all_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' % (
                            args.data_path, args.dataset, 'temp', (i+1)))
                        trn_cur_cld_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (
                            args.data_path, args.dataset, 'temp', (i+1)))
                        tst_cur_cld_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                            args.data_path, args.dataset, 'temp', (i+1)))
                        parameter_drifts = np.load(
                            '%sModel/%s/%s/%d_hist_params_diffs.npy' % (args.data_path, args.dataset, 'temp', i+1))
                        clnt_params_list = np.load(
                            '%sModel/%s/%s/%d_clnt_params_list.npy' % (args.data_path, args.dataset, 'temp', i+1))



    if (not os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt' %(args.data_path, args.dataset, 'temp', args.max_communication_rounds))):

        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        
        else:            
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        
    
        for i in range(saved_itr+1, args.max_communication_rounds):
            inc_seed = 0
            rand_seed = 23
            alpha_coef = 1e-2

            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device) #Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            delta_g_sum = np.zeros(n_par)
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                local_update_last = state_gadient_diffs[clnt] # delta theta_i
                global_update_last = state_gadient_diffs[-1]/weight_list[clnt] #delta theta
                alpha = alpha_coef / weight_list[clnt] 
                hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device) #h_i
                clnt_models[clnt] = train_model_FedDC(model, model_func, alpha,local_update_last, global_update_last,global_mdl, hist_i, 
                                                    trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), 
                                                    batch_size, epoch, print_per, weight_decay, args.dataset, clnt, sch_step=1, sch_gamma=1)


                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par-cld_mdl_param
                parameter_drifts[clnt] += delta_param_curr 
                beta = 1/n_minibatch/learning_rate
                
                state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
                delta_g_cur = (state_g - state_gadient_diffs[clnt])*weight_list[clnt] 
                delta_g_sum += delta_g_cur
                state_gadient_diffs[clnt] = state_g 
                clnt_params_list[clnt] = curr_model_par 
                

                        
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis = 0)
            delta_g_cur = 1 / n_clnt * delta_g_sum  
            state_gadient_diffs[-1] += delta_g_cur  
            
            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)
            
            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)
            all_model     = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))
            
            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 

            
            cent_x = np.concatenate(clnt_x, axis=0)
            cent_y = np.concatenate(clnt_y, axis=0)


            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                            avg_model_sel, args.dataset, 0)
            print("**** Cur Sel Communication %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                %(i+1, acc_tst, loss_tst))
            trn_sel_clt_perf[i] = [loss_tst, acc_tst]
            
            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                            all_model, args.dataset, 0)
            print("**** Cur All Communication %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                %(i+1, acc_tst, loss_tst))
            trn_all_clt_perf[i] = [loss_tst, acc_tst]
            
            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                            cur_cld_model, args.dataset, 0)
            print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                %(i+1, acc_tst, loss_tst))
            trn_cur_cld_perf[i] = [loss_tst, acc_tst]
            
                        
            
            #####

            loss_tst, acc_tst = get_acc_loss(test_x, test_y, 
                                            avg_model_sel, args.dataset, 0)
            print("**** Cur Sel Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                %(i+1, acc_tst, loss_tst))
            tst_sel_clt_perf[i] = [loss_tst, acc_tst]
            
            loss_tst, acc_tst = get_acc_loss(test_x, test_y, 
                                            all_model, args.dataset, 0)
            print("**** Cur All Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                %(i+1, acc_tst, loss_tst))
            tst_all_clt_perf[i] = [loss_tst, acc_tst]
            
            #####

            loss_tst, acc_tst = get_acc_loss(test_x, test_y, 
                                            cur_cld_model, args.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                %(i+1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]
            

            # add to wandb
            wandb.log({
                'Loss/train': {
                    'Sel clients': trn_sel_clt_perf[i][0],
                    'All clients': trn_all_clt_perf[i][0],
                    'Current cloud': trn_cur_cld_perf[i][0]
                },
                'Accuracy/train': {
                    'Sel clients': trn_sel_clt_perf[i][1],
                    'All clients': trn_all_clt_perf[i][1],
                    'Current cloud': trn_cur_cld_perf[i][1]
                },
                'Loss/train_wd': {
                    'Sel clients': get_acc_loss(cent_x, cent_y, avg_model_sel, args.dataset, weight_decay)[0],
                    'All clients': get_acc_loss(cent_x, cent_y, all_model, args.dataset, weight_decay)[0],
                    'Current cloud': get_acc_loss(cent_x, cent_y, cur_cld_model, args.dataset, weight_decay)[0]
                },
                'Loss/test': {
                    'Sel clients': tst_sel_clt_perf[i][0],
                    'All clients': tst_all_clt_perf[i][0],
                    'Current cloud': tst_cur_cld_perf[i][0]
                },
                'Accuracy/test': {
                    'Sel clients': tst_sel_clt_perf[i][1],
                    'All clients': tst_all_clt_perf[i][1],
                    'Current cloud': tst_cur_cld_perf[i][1]
                }
            }, step=i)

            
            if ((i+1) % save_period == 0):
                torch.save(avg_model_sel.state_dict(), '%sModel/%s/%s/ins_avg_%dcom.pt' 
                        %(args.data_path, args.dataset, 'temp', (i+1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/all_avg_%dcom.pt' 
                        %(args.data_path, args.dataset, 'temp', (i+1)))
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt' 
                        %(args.data_path, args.dataset, 'temp', (i+1)))
                
                
                np.save('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' %(args.data_path, args.dataset, 'temp', (i+1)), trn_sel_clt_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' %(args.data_path, args.dataset, 'temp', (i+1)), tst_sel_clt_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' %(args.data_path, args.dataset, 'temp', (i+1)), trn_all_clt_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' %(args.data_path, args.dataset, 'temp', (i+1)), tst_all_clt_perf[:i+1])

                np.save('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' %(args.data_path, args.dataset, 'temp', (i+1)), trn_cur_cld_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(args.data_path, args.dataset, 'temp', (i+1)), tst_cur_cld_perf[:i+1])
                    
                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' %(args.data_path, args.dataset, 'temp', (i+1)), parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(args.data_path, args.dataset, 'temp', (i+1)), clnt_params_list)
                    

                if (i+1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(args.data_path, args.dataset, 'temp', i+1-save_period))

                    
                    
            if ((i+1) % save_period == 0):
                avg_ins_mdls[i//save_period] = avg_model_sel
                avg_all_mdls[i//save_period] = all_model
                avg_cld_mdls[i//save_period] = cur_cld_model
        
        wandb.finish()            
    return avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf




def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--model_name", type = str, default="Resnet18",  help="Basic Name of this run with detailed network-architecture selection. ")
    parser.add_argument("--FL_platform", type = str, default="ViT-FedAVG", choices=[ "Swin-FedAVG", "ViT-FedAVG", "Swin-FedAVG", "EfficientNet-FedAVG", "ResNet-FedAVG"],  help="Choose of different FL platform. ")
    parser.add_argument("--dataset", choices=["cifar10", "Retina"], default="cifar10", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")

    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")
    parser.add_argument("--cfg",  type=str, default="configs/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to args file for Swin-FL',)

    parser.add_argument('--Pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/swin_tiny_patch4_window7_224.pth", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=32, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='2', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization") #99999

    # section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # parser.add_argument("--learning_rate", default=3e-2, type=float, choices=[5e-4, 3e-2, 1e-3],  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # 1e-5 for ViT central

    # FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=10, choices=[10, -1], type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--num_classes", default=10, type=int, help="Num of clasess.")

    args = parser.parse_args()

    train(args)
#exit(0)



if __name__ == "__main__":
    main()



