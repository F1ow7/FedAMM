
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # save file 
    parser.add_argument('--save', type=str, default='save',
                        help="dic to save results (ending without /)")
    parser.add_argument('--init', type=str, default='None',
                        help="location of init model")
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--malicious',type=float,default=0, help="proportion of mailicious clients")

    parser.add_argument('--attack_type', type=str, default='untarget')
    
    #***** badnet labelflip layerattack updateflip get_weight layerattack_rev layerattack_ER****
    parser.add_argument('--attack', type=str,
                        default='badnet', help='attack method')
    
    parser.add_argument('--poison_frac', type=float, default=0.2, 
                        help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")

    # *****local_ep = 3, local_bs=50, lr=0.1*******
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")

    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")

    # model arguments
    #*************************model******************************#
    # resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist
    parser.add_argument('--model', type=str,
                        default='Mnist_CNN', help='model name')

    # other arguments
    #*************************dataset*******************************#
    # fashion_mnist mnist cifar
    parser.add_argument('--dataset', type=str,
                        default='mnist', help="name of dataset")
    

    # OPTICS cluster arguments
    parser.add_argument('--cluster_method', type=str,
                        default='dbscan', help="cluster method")
    parser.add_argument('--min_samples', type=int,
                        default=2, help="min samples")
    parser.add_argument('--min_cluster_size', type=int,
                        default=3, help="min cluster size")                    
    
    

    parser.add_argument('--defence', type=str,
                        default='avg', help="strategy of defence")
    # parser.add_argument('--iid', action='store_true',
    #                     help='whether i.i.d or not')
    parser.add_argument('--iid', type=int, default=1,
                        help='whether i.i.d or not')

 #************************atttack_label********************************#
    parser.add_argument('--attack_label', type=int, default=5,
                        help="trigger for which label")
    # attack_goal=-1 is all to one
    parser.add_argument('--attack_goal', type=int, default=7,
                        help="trigger to which label")
    # --attack_begin 70 means accuracy is up to 70 then attack
    parser.add_argument('--attack_begin', type=int, default=0,
                        help="the accuracy begin to attack")
    
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--robustLR_threshold', type=int, default=4, 
                        help="break ties when votes sum to 0")
    
    parser.add_argument('--server_dataset', type=int,default=200,help="number of dataset in server")
    
    parser.add_argument('--server_lr', type=float,default=1,help="number of dataset in server using in fltrust")
    
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.5)")
    

    parser.add_argument('--normalization', type=str, default='maxmin')
    
    parser.add_argument('--split', type=str, default='user',
                        help="train-test split type, user or sample")   
    #*********trigger info*********
    #  square  apple  watermark  
    parser.add_argument('--trigger', type=str, default='square',
                        help="Kind of trigger")  
    # mnist 28*28  cifar10 32*32
    parser.add_argument('--triggerX', type=int, default='0',
                        help="position of trigger x-aix") 
    parser.add_argument('--triggerY', type=int, default='0',
                        help="position of trigger y-aix")
    
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=3322,
                        help='random seed (default: 3322)')
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    
    parser.add_argument('--mal_score', type=float, default=0)
    parser.add_argument('--ben_score', type=float, default=0)
    
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)
    parser.add_argument('--all_clients', action='store_true',
                        help='aggregation over all clients') 
    
    parser.add_argument('--add_noise',type=int, default=1)
    parser.add_argument('--k_coeff', type=float, default=0.01)

    parser.add_argument('--layer_por',type=float,default=0.5)

    parser.add_argument('--adv', type=str, default='ada',
                        help="ada or sta")  
    parser.add_argument('--q',type=str, default='1' )

    parser.add_argument('--tn', type=int,default=0)
    parser.add_argument('--fn', type=int,default=0)

    parser.add_argument('--tp', type=int,default=0)
    parser.add_argument('--fp', type=int,default=0)

    args = parser.parse_args()
    return args
