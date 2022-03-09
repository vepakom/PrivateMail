import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity

import torch

import matplotlib.pyplot as plt
import seaborn as sns 
from colour import Color

from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

import torch.nn.functional as f

import os


def laplacian(A):
    '''Returns graph Laplacian given an adjacency matrix.'''
    D = A.sum(dim=0)
    return torch.diag(D) - A

def umeyama(P, Q):
    '''
    Kabsch-Umeyama rigidly (+scale) aligns two point clouds with known point-to-point correspondences.
    
    Parameters:
        P (ndarray): Point cloud being aligned.
        Q (ndarray): Target point cloud.

    Returns:
        c (float): Constant scale factor.
        R (ndarray): Rotation matrix.
        t (ndarray): Translation matrix.
    '''
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


class DPImageRetriever:
    '''
    DP image retriever.
    '''
    def __init__(self, data_base, embed_params, noise_params={}, retrieval_num=8, n_public=1000, n_gallery=-1):
        self.data_base = data_base
        self.embed_params = embed_params
        self.noise_params = noise_params
        self.retrieval_num = retrieval_num
        
        self.c = len(torch.unique(data_base['test_labels'])) # Num classes
        
        self.colors = np.array(list(Color("red").range_to(Color("purple"),len(np.unique(data_base['train_labels'].cpu())))))
        color_to_hex = lambda c: c.hex_l
        self.func = np.vectorize(color_to_hex)
        
        # Public data
        self.public_labels = torch.as_tensor(data_base['train_labels'], dtype=torch.float32).squeeze()
        idx = torch.randperm(self.public_labels.shape[0])[:n_public]
        self.public_labels = self.public_labels[idx]
        public_features = data_base['train_features']
        self.public_features = torch.as_tensor(normalize(public_features.cpu(), axis=1), dtype=torch.float32)[idx].cuda()
        self.public_colors = self.func(self.colors[list(data_base['train_labels'][idx].cpu().numpy())])
        
        # Server data
        self.gallery_labels = torch.as_tensor(data_base['test_labels'], dtype=torch.float32).cuda()
        idx = torch.randperm(self.gallery_labels.shape[0])[:n_gallery]
        self.gallery_labels = self.gallery_labels[idx]
        gallery_features = data_base['test_features'][idx]
        self.gallery_features = torch.as_tensor(normalize(gallery_features.cpu(), axis=1), dtype=torch.float32).cuda()
        self.gallery_colors = self.func(self.colors[list(data_base['test_labels'][idx].cpu().numpy())])
        self.gallery_images = data_base['test_images'][idx]
        
        self.init_Q()
        
        # Pre-compute server embedding
        self.server_embedding = self.server_embed()  
        
    def set_embed_params(self, **kwargs):
        for key, val in kwargs.items():
            self.embed_params[key] = val
            
        self.init_Q()
        
        # Re-compute server embedding
        self.server_embedding = self.server_embed()
        
    def set_noise_params(self, **kwargs):
        for key, val in kwargs.items():
            self.noise_params[key] = val
            
    def init_Q(self):
        ''' Random initialized Q '''
        sigma_q = self.embed_params['sigma_q']
        dim = self.embed_params['dim']
        
        client_n = len(self.public_labels)+1
        self.client_Q = torch.normal(mean=torch.tensor(0), std=torch.tensor(sigma_q), size=(client_n, dim)).cuda()
        
        server_n = len(self.gallery_labels)+len(self.public_labels)
        self.server_Q = torch.normal(mean=torch.tensor(0), std=torch.tensor(sigma_q), size=(server_n, dim)).cuda()
        
    def get_query_data(self, query_idx):
        query_label = torch.as_tensor(self.gallery_labels[query_idx]).cuda()
        query_feature = self.gallery_features[query_idx]
        query_feature = torch.as_tensor(normalize(query_feature.cpu().reshape(1, -1)), dtype=torch.float32).cuda()
        query_color = self.func(self.colors[self.data_base['test_labels'][query_idx].item()])
    
        return query_feature, query_label, query_color
    
    def client_embed(self, query_feature, query_label, query_color, noise=False):
        # Anchor query with public data
        client_features = torch.cat((query_feature, self.public_features))
        client_labels = torch.cat((query_label[None, None], self.public_labels[:, None]))
        client_colors = np.concatenate((query_color[None], self.public_colors))

        return self.embed(client_features, client_labels, self.client_Q, noise=noise, **self.embed_params), client_colors
    
    def server_embed(self):
        # Anchor gallery with public
        server_features = torch.cat((self.gallery_features, self.public_features), dim=0)
        server_labels = torch.cat((self.gallery_labels[:, None], self.public_labels[:, None]), dim=0)

        return self.embed(server_features, server_labels, self.server_Q, noise=False, **self.embed_params)
    
    def embed(self, X, Y, Q, noise, sigma, alpha, dim, sigma_q, iters):
        '''
        Returns an n by dim geometric embedding for X, Y.
        If noise=True, adds noise according to the PrivateMail scheme to guarantee privacy.
        '''
        n = len(Y)
        X = f.normalize(X, p=2, dim=1) # Normalization
        
        def get_LX(X, sigma):
            gamma = 1/(sigma**2) # RBF kernel width
            X_kernel = rbf_kernel(X.cpu().numpy(), gamma=gamma)
            np.fill_diagonal(X_kernel, 0)
            return laplacian(torch.as_tensor(X_kernel)).cuda()

        def get_LY(Y):
            dist_Y = torch.cdist(Y, Y).cuda()
            n = len(Y) 
            I_X = torch.eye(n).cuda()
            J = I_X - 1/n * torch.ones(n, n).cuda() 
            Y_kernel = -0.5 * J @ dist_Y @ J
            return laplacian(Y_kernel) 
        
        # Calculate Laplacians
        LX = get_LX(X, sigma)
        LY = get_LY(Y)
        
        diag_LX_inv = 1/(torch.diag(LX) + 1e-9) #Avoid division by 0
        
        embedding = Q #Initial X
        
        if noise:
            embedding = (0.5*torch.diag(diag_LX_inv) @ (alpha*LY - LX) @ embedding) + embedding #Embedding update
            assert self.noise_params, "No noise parameters defined."

            sens = self.sensitivity(n, **self.embed_params, **self.noise_params)
            epsilon, delta = self.noise_params['epsilon'], self.noise_params['delta']

            #Gaussian noise mechanism
            scale = 2*np.log(1.25/delta) * sens**2/(epsilon**2)
            noise_mtx = np.random.normal(scale = scale, size=embedding.shape)

            embedding += torch.as_tensor(noise_mtx).cuda()
                
            # Recompute Gaussian kernel
            LX = get_LX(embedding, sigma)
            diag_LX_inv = 1/(torch.diag(LX) + 1e-9)
        
        for i in range(iters):
            embedding = (0.5*torch.diag(diag_LX_inv) @ (alpha*LY - LX) @ embedding) + embedding 
            embedding = np.sqrt(n)*embedding/torch.linalg.norm(embedding, ord='fro') #Avoid convergence to one point
            
        return embedding
    
    def sensitivity(self, n, sigma, alpha, dim, sigma_q, **kwargs):
        '''Computes global sensitivity'''
        c = self.c - 1 # Max integer label
        zii_max = alpha**2*(n/(n*np.exp(-2/(sigma**2))+np.exp(-1/(2*sigma**2))-1))**2 \
                        + alpha**2 * (n/((n+1)*np.exp(-2/(sigma**2))-1))**2 \
                        - 2*alpha**2*((n+1)*np.exp(-c/(2*sigma**2))-1)**2/(n*(n+np.exp(-1/(2*sigma**2))-1))
 
        zij_max = (alpha**2+1)/((n*np.exp(-2/(sigma**2))+np.exp(-1/(2*sigma**2))-1)**2) \
                    - (2*alpha*np.exp(-(c**2+4)/(2*sigma**2)))/((n+np.exp(-1/(2*sigma**2))-1)**2) \
                    + (alpha**2+1)/(((n+1)*np.exp(-2/(sigma**2))-1)**2) \
                    - 2*alpha*np.exp(-(c**2+4)/(2*sigma**2))/(n**2) \
                    - 2*(alpha**2*np.exp(-(c**2)/(sigma**2))+np.exp(-4/(sigma**2)))/(n*(n+np.exp(-1/(2*sigma**2))-1)) \
                    + 4*alpha/((n*np.exp(-2/(sigma**2))+np.exp(-1/(2*sigma**2))-1)*((n+1)*np.exp(-2/(sigma**2))-1))
        
        phi_max = n*zij_max + zii_max
       
        sens = 0.5*phi_max*np.sqrt(n+1)*torch.linalg.norm(self.client_Q, ord='fro').cpu() 
        return sens
    
    def recall(self, k, noise=False, verbose=False):
        '''Returns recall@k.'''
        total_correct = 0
        n = len(self.gallery_labels)
        desc = 'Computing recall@'+str(k)
        if noise:
            assert self.noise_params, "No noise parameters defined."
            desc += ' for \u03B5 = '+str(self.noise_params['epsilon']) + ', \u03B4 = '+str(self.noise_params['delta'])
        pbar = tqdm(enumerate(self.gallery_images), total=n, desc=desc)
        for query_idx, query_img_name in pbar:
            retrieved_labels = self.run_query(query_img_name, k, noise=noise, save=False)[1]
            query_label = self.get_query_data(query_idx)[1]
            query_label = int(query_label.item())
            total_correct += (query_label in retrieved_labels)
        recall = total_correct/n
        if verbose:
            print('Recall:', recall, flush=True)
        return recall
    
    def run_query(self, query_img_name, k, noise=False, verbose=False, save=False):
        """
        Runs Differentially Private Image Retrieval for k images using Private-Mail subroutine.
        
        Returns:
            retrieved_img_names (list): Retrieved image file names.
            retrieved_labels (list): Integer labels for retrieved images.
       
        If verbose = True, prints labels for the retrieved images.
        If save = True, saves retrieved image files.
        """
        query_idx = np.where(self.gallery_images == query_img_name)[0][0]
        query_feature, query_label, query_color = self.get_query_data(query_idx)
        client_embedding, client_colors = self.client_embed(query_feature, query_label, query_color, noise=noise)
        client_embedding = client_embedding.cpu()
        server_embedding = self.server_embedding.cpu() # pre-computed

        # Align public from client and server
        c, R, t = umeyama(client_embedding[1:].numpy(), 
                          server_embedding[len(self.gallery_labels):].numpy())
        
        # Apply alignment params to entire client embedding
        client_embedding = client_embedding@(c*R) + t 

        query_feature = torch.as_tensor(client_embedding[:1]).double()
        gallery_features = server_embedding[:len(self.gallery_labels)].double()
        query_label = int(query_label.item())
        retrieved_img_names, retrieved_labels = self.retrieve(k, query_idx, query_img_name, query_label, 
                                                              query_feature, gallery_features, save)
        if verbose:
            print('Query image:', query_img_name)
            print('Query label:', query_label)
            print('Retrieved labels:', retrieved_labels)
        
        return retrieved_img_names, retrieved_labels
        
            
    def retrieve(self, k, query_idx, query_img_name, query_label, query_feature, gallery_features, save):
        """
        Finds top k image matches based on distances between query feature and gallery features.
        
        Returns:
            retrieved_img_names (list): Retrieved image file names.
            retrieved_labels (list): Integer labels for retrieved images.
            
        If save = True, saves retrieved image files.
        """
        dist_matrix = torch.cdist(query_feature.cuda(), gallery_features.cuda()).squeeze()
        dist_matrix[query_idx] = float('inf')
        idxs = dist_matrix.topk(k=k, dim=-1, largest=False)[1]

        if save:
            query_img = Image.open(query_img_name).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
            result_path = 'results/{}'.format(query_img_name.split('/')[-1].split('.')[0])
            if os.path.exists(result_path):
                shutil.rmtree(result_path)
            os.mkdir(result_path)
            query_img.save('{}/query_img.jpg'.format(result_path))
        
        retrieved_img_names = []
        retrieved_labels = []
        for num, idx in enumerate(idxs):
            retrieved_img_name = self.gallery_images[idx.item()]
            retrieved_img_names.append(retrieved_img_name)
            retrieved_label = int(self.gallery_labels[idx.item()].item())
            retrieved_labels.append(retrieved_label)
            if save: 
                retrieved_img = Image.open(retrieved_img_name).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
                draw = ImageDraw.Draw(retrieved_img)
                if query_label == retrieved_label:
                    draw.rectangle((0, 0, 223, 223), outline='green', width=8)
                else:
                    draw.rectangle((0, 0, 223, 223), outline='red', width=8)
                retrieved_img.save('{}/retrieved_img_{}.jpg'.format(result_path, num + 1))
        return retrieved_img_names, retrieved_labels 
   