
import numpy as np
import torch
import copy
import time
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import torch.nn.functional as F
import scipy.stats

def cos(a, b):
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res



def weighted_aggregation_del_min(params, global_parameters, score_list,min_index):
    total_num = len(params)
    sum_parameters = None
    total_score = sum(score_list)-score_list[min_index]
    for i in range(total_num):
        if i == min_index:
            continue
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = score_list[i]*var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + score_list[i]*params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_score)

    return global_parameters

def MaxMinNormalization(list):
    new_list=[]
    for x in list:
        x = (x - min(list)) / (max(list) - min(list))
        new_list.append(x)
    return new_list

def ZScoreNormalization(list):
    new_list=[]
    for x in list:
        x = (x - np.average(list)) / np.std(list)
        new_list.append(x)
    return new_list

def sigmoid(list):
    new_list=[]
    for x in list:
        x = 1.0 / (1+np.exp(-float(x)))
        new_list.append(x)
    return new_list

def filter_dct(dct, threshold):
    n= list(dct.shape)
    i_threshold = int(n[0] * threshold)
    sorted_dct = torch.sort(torch.abs(dct.view(-1)))[0]
    threshold_value = sorted_dct[i_threshold]
    filtered_dct = torch.where(torch.abs(dct) <= threshold_value, dct, torch.zeros_like(dct))
    return filtered_dct

def get_weight(model_weight):
    weight_tensor_result = []
    for k, v in model_weight.items():
        weight_tensor_result.append(v.reshape(-1).float())
    weights = torch.cat(weight_tensor_result)
    return weights
    

def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)



def get_update_static(nets_this_round, global_net):
    model_weight_list = []
    global_weight = get_weight(global_net).unsqueeze(0)

    for i in range(len(nets_this_round)):
        net_para = nets_this_round[i]
        model_weight = get_weight(net_para).unsqueeze(0)
        model_update = model_weight - global_weight
        model_weight_list.append(model_update)
    model_weight_cat = torch.cat(model_weight_list, dim=0)
    model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)
    
    return model_mean, model_std, model_weight_cat, global_weight


def get_foolsgold(grads, global_weight):
    n_clients = grads.shape[0]
    grads_norm = F.normalize(grads, dim=1)
    cs = torch.mm(grads_norm, grads_norm.T)
    cs = cs.cpu() - torch.eye(n_clients)
    maxcs, _ = torch.max(cs, axis=1)

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    maxcs_2, _ = torch.max(cs, axis=1) 
    wv = 1 - maxcs_2
    

    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / torch.max(wv)
    wv[(wv == 1)] = .99
       
    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    
    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i]*grads[i] 
            model_weight_list.append(current_weight)
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)
        
    return fools_gold_weight.view(-1), wv




def get_foolsgold_score(total_score, grads, global_weight):
    n_clients = total_score.shape[0]
    norm_score = total_score
    
    wv = (norm_score - np.min(norm_score)) / (np.max(norm_score) - np.min(norm_score))
    wv[(wv == 1)] = .99
    wv[(wv == 0)] = .01
       
    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    model_weight_list = []
    for i in range(0, n_clients):
        if wv[i] != 0:
            current_weight = global_weight + wv[i] * grads[i] 
            model_weight_list.append(current_weight)
    fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)
        
    return fools_gold_weight.view(-1), wv



def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters

def weighted_aggregation(params, global_parameters, score_list):
    total_num = len(params)
    sum_parameters = None
    total_score = sum(score_list)
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = score_list[i]*var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + score_list[i]*params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_score)

    return global_parameters




def update_aggregate(params):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in sum_parameters:
        sum_parameters[var] = sum_parameters[var] / total_num
    return sum_parameters



def multi_krum(gradients, n_attackers, args, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        print(scores)
        args.krum_distance.append(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    args.turn+=1
    if multi_k == False:
        if candidate_indices[0] < num_malicious_clients:
            args.wrong_mal += 1
            
    print(candidate_indices)
    
    print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2



def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    # args.robustLR_threshold = 6
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   
    


def flame(local_model, update_params, global_model, args):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))

    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)



    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN

   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1
    print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
    print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model
    

def fedcpa(local_model, update_params, global_model,prev_global_w, args):
    local_global_w_list = []
    global_critical_dict = {}
    for name, val in global_model.items():
        if val.dim() in [2, 4]:
            critical_weight = torch.abs((global_model[name] - prev_global_w[name]) * global_model[name])
            global_critical_dict[name] = critical_weight
    
    global_w_stacked = get_weight(global_critical_dict).view(1, -1)      
    global_topk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * 0.01)).indices
    global_bottomk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * 0.01), largest=False).indices


    for i in range(len(local_model)):
        net_para = local_model[i]
        critical_dict = {}
        for name, val in net_para.items():
            if val.dim() in [2, 4]:
                critical_weight = torch.abs(update_params[i][name] * val)
                critical_dict[name] = critical_weight
            
        local_global_w_list.append(get_weight(critical_dict))
           
    w_stacked = torch.stack(local_global_w_list, dim=0)
    local_topk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * 0.01)).indices
    local_bottomk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * 0.01), largest=False).indices



    pairwise_score = np.zeros((len(local_model), len(local_model)))
    for i in range(len(local_model)):
        for j in range(len(local_model)):
            if i == j:
                pairwise_score[i][j] = 1
            elif i < j:
                continue       
            topk_intersection = list(set(local_topk_indices[i].tolist()) & set(local_topk_indices[j].tolist()))
            topk_pearsonr = scipy.stats.pearsonr(w_stacked[i, topk_intersection].cpu().numpy(), w_stacked[j, topk_intersection].cpu().numpy())[0]
            if np.isnan(topk_pearsonr):
                topk_pearsonr = 0
            topk_corr_dist = (topk_pearsonr + 1) / 2
            topk_jaccard_dist = len(topk_intersection) / (len(local_topk_indices[i]) + len(local_topk_indices[j]) - len(topk_intersection))

            bottomk_intersection = list(set(local_bottomk_indices[i].tolist()) & set(local_bottomk_indices[j].tolist()))
            bottom_pearsonr = scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].cpu().numpy(), w_stacked[j, bottomk_intersection].cpu().numpy())[0]
            if np.isnan(bottom_pearsonr):
                bottom_pearsonr = 0
            bottomk_corr_dist = (bottom_pearsonr + 1) / 2       
            bottomk_jaccard_dist = len(bottomk_intersection) / (len(local_bottomk_indices[i]) + len(local_bottomk_indices[j]) - len(bottomk_intersection))        

            pairwise_score[i][j] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2
            pairwise_score[j][i] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2

    global_score = np.zeros(len(local_model))
    for i in range(len(local_model)):

        topk_intersection = list(set(local_topk_indices[i].tolist()) & set(global_topk_indices[0].tolist()))
        topk_pearsonr = scipy.stats.pearsonr(w_stacked[i, topk_intersection].cpu().numpy(), global_w_stacked[0, topk_intersection].cpu().numpy())[0]
        if np.isnan(topk_pearsonr):
            topk_pearsonr = 0
        topk_corr_dist = (topk_pearsonr + 1) / 2
        topk_jaccard_dist = len(topk_intersection) / (len(local_topk_indices[i]) + len(global_topk_indices[0]) - len(topk_intersection))

        bottomk_intersection = list(set(local_bottomk_indices[i].tolist()) & set(global_bottomk_indices[0].tolist()))
        bottom_pearsonr=scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].cpu().numpy(), global_w_stacked[0, bottomk_intersection].cpu().numpy())[0]
        if np.isnan(bottom_pearsonr):
            bottom_pearsonr = 0
        bottomk_corr_dist = (bottom_pearsonr + 1) / 2 
        bottomk_jaccard_dist = len(bottomk_intersection) / (len(local_bottomk_indices[i]) + len(global_bottomk_indices[0]) - len(bottomk_intersection))        

        global_score[i]= (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2
            
    total_score = np.mean(pairwise_score, axis=1) + global_score


    update_mean, update_std, update_cat, global_weight = get_update_static(local_model, global_model)
    model_weight_foolsgold, wv  = get_foolsgold_score(total_score, update_cat, global_weight)

    
    current_idx = 0
    for key in global_model:
        length = len(global_model[key].reshape(-1))
        global_model[key] = model_weight_foolsgold[current_idx:current_idx+length].reshape(global_model[key].shape)
        current_idx += length


    return global_model



def foolsgold(local_model, global_model, args):
    model_weight_list = []

    for i in range(len(local_model)):
        net_para = local_model[i]
        model_weight = get_weight(net_para).unsqueeze(0)
        model_weight_list.append(model_weight)

    model_weight_cat= torch.cat(model_weight_list, dim=0)
    update_mean, update_std, update_cat, global_weight = get_update_static(local_model, global_model)
    model_weight_foolsgold, wv  = get_foolsgold(update_cat, global_weight)
    
    
    current_idx = 0
    for key in global_model:
        length = len(global_model[key].reshape(-1))
        global_model[key] = model_weight_foolsgold[current_idx:current_idx+length].reshape(global_model[key].shape)
        current_idx +=length 

    return global_model



def fedmm(local_model, update_params, global_model,prev_global_w, args):


    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    update_params_vector = []



    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))

    for param in update_params:
        update_params_vector.append(parameters_dict_to_vector(param))


    PCA_local_model = torch.stack(local_model_vector)
    pca = PCA(n_components=0.9,svd_solver='full')
    pca.fit(PCA_local_model.cpu())
    transformed_local_model = pca.transform(PCA_local_model.cpu())
    PCA_local_model_vector = transformed_local_model.tolist()


    

    for i in range(len(PCA_local_model_vector)):
        cos_i = []
        for j in range(len(PCA_local_model_vector)):
            cos_ij = 1- min(cos(torch.tensor(PCA_local_model_vector[i]),torch.tensor(PCA_local_model_vector[j])),torch.tensor(1))
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)


    clusterer = OPTICS(min_samples=args.min_samples,metric='precomputed',cluster_method=args.cluster_method,min_cluster_size=args.min_cluster_size).fit(cos_list)

    
    
    cluster_list = []
    cluster_score_list = []
    proxy_model_list = []
    norm_list = np.array([])
    proxy_norm_list = np.array([])


    if clusterer.labels_.max() < 0:
        temp_client = []
        for i in range(len(local_model)):
            temp_client.append(i)
        cluster_list.append(temp_client)
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            temp_client = []
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == index_cluster:
                    temp_client.append(i)
            cluster_list.append(temp_client)
    
    for cluster in cluster_list:
        proxy_model = update_aggregate([update_params[i] for i in cluster])
        proxy_model_list.append(proxy_model)



    local_global_w_list = []
    global_critical_dict = {}
    for name, val in global_model.items():
        if val.dim() in [2, 4]:
            critical_weight = torch.abs((global_model[name] - prev_global_w[name]) * global_model[name])
            global_critical_dict[name] = critical_weight
    global_w_stacked = get_weight(global_critical_dict).view(1, -1)      
    global_topk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * args.k_coeff)).indices
    global_bottomk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * args.k_coeff), largest=False).indices


    for i in range(len(local_model)):
        net_para = local_model[i]
        critical_dict = {}
        for name, val in net_para.items():
            if val.dim() in [2, 4]:
                critical_weight = torch.abs(update_params[i][name] * val)
                critical_dict[name] = critical_weight
            
        local_global_w_list.append(get_weight(critical_dict))
           
    w_stacked = torch.stack(local_global_w_list, dim=0)


    local_topk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * args.k_coeff)).indices
    local_bottomk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * args.k_coeff), largest=False).indices


    pairwise_score = np.zeros((len(local_model), len(local_model)))
    for i in range(len(local_model)):
        for j in range(len(local_model)):
            if i == j:
                pairwise_score[i][j] = 1
            elif i < j:
                continue       
            topk_intersection = list(set(local_topk_indices[i].tolist()) & set(local_topk_indices[j].tolist()))
            topk_pearsonr = scipy.stats.pearsonr(w_stacked[i, topk_intersection].cpu().numpy(), w_stacked[j, topk_intersection].cpu().numpy())[0]
            if np.isnan(topk_pearsonr):
                topk_pearsonr = 0
            topk_corr_dist = (topk_pearsonr + 1) / 2
            topk_jaccard_dist = len(topk_intersection) / (len(local_topk_indices[i]) + len(local_topk_indices[j]) - len(topk_intersection))

            bottomk_intersection = list(set(local_bottomk_indices[i].tolist()) & set(local_bottomk_indices[j].tolist()))
            bottom_pearsonr = scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].cpu().numpy(), w_stacked[j, bottomk_intersection].cpu().numpy())[0]
            if np.isnan(bottom_pearsonr):
                bottom_pearsonr = 0
            bottomk_corr_dist = (bottom_pearsonr + 1) / 2       
            bottomk_jaccard_dist = len(bottomk_intersection) / (len(local_bottomk_indices[i]) + len(local_bottomk_indices[j]) - len(bottomk_intersection))        

            pairwise_score[i][j] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2
            pairwise_score[j][i] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2

    global_score = np.zeros(len(local_model))
    for i in range(len(local_model)):
        topk_intersection = list(set(local_topk_indices[i].tolist()) & set(global_topk_indices[0].tolist()))
        topk_pearsonr = scipy.stats.pearsonr(w_stacked[i, topk_intersection].cpu().numpy(), global_w_stacked[0, topk_intersection].cpu().numpy())[0]
        if np.isnan(topk_pearsonr):
            topk_pearsonr = 0
        topk_corr_dist = (topk_pearsonr + 1) / 2
        topk_jaccard_dist = len(topk_intersection) / (len(local_topk_indices[i]) + len(global_topk_indices[0]) - len(topk_intersection))

        bottomk_intersection = list(set(local_bottomk_indices[i].tolist()) & set(global_bottomk_indices[0].tolist()))
        bottom_pearsonr=scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].cpu().numpy(), global_w_stacked[0, bottomk_intersection].cpu().numpy())[0]
        if np.isnan(bottom_pearsonr):
            bottom_pearsonr = 0
        bottomk_corr_dist = (bottom_pearsonr + 1) / 2 
        bottomk_jaccard_dist = len(bottomk_intersection) / (len(local_bottomk_indices[i]) + len(global_bottomk_indices[0]) - len(bottomk_intersection))        

        global_score[i]= (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2

    
    total_score = np.negative(np.mean(pairwise_score, axis=1)+ global_score)



    for cluster in cluster_list:
        temp_score = 0
        for i in cluster:
            temp_score += total_score[i]
        temp_score = temp_score / len(cluster)
        cluster_score_list.append(temp_score)

    if len(cluster_score_list)>1:
        if args.normalization == 'maxmin':
            norm_cluster_score_list=MaxMinNormalization(cluster_score_list)
        elif args.normalization == 'zscore':
            norm_cluster_score_list=sigmoid(ZScoreNormalization(cluster_score_list))
    else:
        norm_cluster_score_list=[1.0]


    max_index=norm_cluster_score_list.index(max(norm_cluster_score_list))

    min_index=norm_cluster_score_list.index(min(norm_cluster_score_list))
    

    for i in range(len(local_model_vector)):
        # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
        norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2).item())  # no consider BN

    for i in range(len(proxy_model_list)):
        proxy_norm_list = np.append(proxy_norm_list,torch.norm(parameters_dict_to_vector(proxy_model_list[i]),p=2).item())  # no consider BN


    clip_value = np.median(norm_list)
    for i in range(len(proxy_model_list)):
        gama = clip_value/proxy_norm_list[i]
        if gama < 1:
            for key in proxy_model_list[i]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                proxy_model_list[i][key] *= gama



 
    join_agg_list=cluster_list[max_index]


    epoch_wrong_mal = 0
    epoch_right_ben = 0

    for i in range(len(join_agg_list)):
        if join_agg_list[i] < num_malicious_clients:
            args.wrong_mal += 1
            epoch_wrong_mal += 1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
            epoch_right_ben += 1


    for i in range(len(join_agg_list)):
        if join_agg_list[i] >= num_malicious_clients:
            args.tn += 1


    args.turn+=1

    args.tn=args.right_ben
    args.fn=args.wrong_mal
    args.fp=num_benign_clients*args.turn-args.tn
    args.tp=num_malicious_clients*args.turn-args.fn

    epoch_tn=epoch_right_ben
    epoch_fn=epoch_wrong_mal
    epoch_fp=num_benign_clients-epoch_tn
    epoch_tp=num_malicious_clients-epoch_fn




    for var in global_model:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_model[var] = proxy_model_list[max_index][var]
            continue
        global_model[var] += proxy_model_list[max_index][var]
        

    layer_avg = {}
    prev_layer_avg = {}
    layer_variation_avg = {}
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = copy.deepcopy(var)
        # layer_avg.append(temp.mean()) 
        layer_avg[key] = temp.mean()
    
    for key, var in prev_global_w.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = copy.deepcopy(var)
        # prev_layer_avg.append(temp.mean())
        prev_layer_avg[key] = temp.mean()


    for key in layer_avg.keys():
        layer_variation_avg[key] = torch.sub(layer_avg[key],prev_layer_avg[key])
    
    sorted_keys = sorted(layer_variation_avg.keys(), key=lambda k:layer_variation_avg[k],reverse=True)

    if args.add_noise == 1:
        for key, var in global_model.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            n = max(1, len(sorted_keys) * 50 // 100)
            if key in sorted_keys[-n:]:
                temp = copy.deepcopy(var)
                temp = temp.normal_(mean=0,std=args.noise*clip_value)
                var += temp    

    return global_model



