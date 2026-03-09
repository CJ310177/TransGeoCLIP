import faiss
import torch
import numpy as np
import os
import argparse
import pandas as pd
import ast
import itertools
from PIL import Image
from geopy.distance import geodesic
from transformers import CLIPImageProcessor, CLIPModel
from utils.utils import MP16Dataset, im2gps3kDataset, yfcc4kDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from utils.TransGeoCLIP import TransGeoCLIP



def build_index(args):
    if args.index == 'TransGeoCLIP':
        model = TransGeoCLIP(device=args.device).to(args.device)
        checkpoint = torch.load('your path', map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.requires_grad_(False)
        vision_processor = model.vision_processor
        dataset = MP16Dataset(vision_processor = model.vision_processor, text_processor = None)
        index_flat = faiss.IndexFlatIP(768*3)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=3)
        model.eval()
        t= tqdm(dataloader)
        for i, (images, texts, longitude, latitude) in enumerate(t):
            images = images.to(args.device)
            vision_output = model.vision_model(images)[1]
            image_embeds = model.vision_projection(vision_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

            image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
            image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

            image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
            image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

            image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
            index_flat.add(image_embeds.cpu().detach().numpy()) # pyright: ignore[reportCallIssue]

        faiss.write_index(index_flat, f'your path/{args.index}.index')

def search_index(args, index, topk):
    print('start searching...')
    if args.dataset == 'im2gps3k':
        if args.index == 'TransGeoCLIP':
            model = TransGeoCLIP(device=args.device).to(args.device)
            checkpoint = torch.load('your path', map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])  # 关键修改            
            model.requires_grad_(False)
            vision_processor = model.vision_processor
            dataset = im2gps3kDataset(vision_processor = vision_processor, text_processor = None)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)
            test_images_embeds = np.empty((0, 768*3))
            model.eval()
            print('generating embeds...')
            t = tqdm(dataloader)
            for i, (images, texts, longitude, latitude) in enumerate(t):
                images = images.to(args.device)
                vision_output = model.vision_model(images)[1]
                image_embeds = model.vision_projection(vision_output)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
                image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

                image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
                image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

                image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
                test_images_embeds = np.concatenate([test_images_embeds, image_embeds.cpu().detach().numpy()], axis=0)
            print(test_images_embeds.shape)
            test_images_embeds = test_images_embeds.reshape(-1, 768*3)
            print('start searching NN...')
            test_images_embeds = np.ascontiguousarray(test_images_embeds.astype(np.float32))
            D, I = index.search(test_images_embeds, topk)
            print(I)
            return D, I
    elif args.dataset == 'yfcc4k':
        if args.index == 'TransGeoCLIP':
            model = TransGeoCLIP(device=args.device).to(args.device)
            checkpoint = torch.load('your path', map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])  # 关键修改            
            model.requires_grad_(False)
            vision_processor = model.vision_processor
            dataset = yfcc4kDataset(vision_processor = vision_processor, text_processor = None)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)
            test_images_embeds = np.empty((0, 768*3))
            model.eval()
            print('generating embeds...')
            t = tqdm(dataloader)
            for i, (images, texts, longitude, latitude) in enumerate(t):
                images = images.to(args.device)
                vision_output = model.vision_model(images)[1]
                image_embeds = model.vision_projection(vision_output)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

                image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
                image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)

                image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
                image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)

                image_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
                test_images_embeds = np.concatenate([test_images_embeds, image_embeds.cpu().detach().numpy()], axis=0)
            print(test_images_embeds.shape)
            test_images_embeds = test_images_embeds.reshape(-1, 768*3)
            test_images_embeds = np.ascontiguousarray(test_images_embeds.astype(np.float32))
            print('start searching NN...')
            D, I = index.search(test_images_embeds, topk)
            return D, I

class GeoImageDataset(Dataset):
    def __init__(self, dataframe, img_folder, topn, vision_processor, database_df, I):
        self.dataframe = dataframe
        self.img_folder = img_folder
        self.topn = topn
        self.vision_processor = vision_processor
        self.database_df = database_df
        self.I = I

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f'{self.img_folder}/{self.dataframe.loc[idx, "IMG_ID"]}'
        image = Image.open(img_path).convert('RGB')
        image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3,224,224)
        
        gps_data = []
        search_top1_latitude, search_top1_longitude = self.database_df.loc[self.I[idx][0], ['LAT', 'LON']].values
        rag_5, rag_10, rag_15, zs = [],[],[],[]
        for j in range(self.topn):
            gps_data.extend([
                float(self.dataframe.loc[idx, f'5_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'5_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'10_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'10_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'15_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'15_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'zs_{j}_latitude']),
                float(self.dataframe.loc[idx, f'zs_{j}_longitude']),
                search_top1_latitude,
                search_top1_longitude
            ])
        
        gps_data = np.array(gps_data).reshape(-1, 2)
        return image, gps_data, idx

def evaluate(args, I):
    print('start evaluation')
    if args.database == 'mp16':
        # 1. 仅使用检索结果I和数据库信息进行评估
        database = args.database_df  # 数据库（MP16）包含真实经纬度
        df = args.dataset_df        # 测试集（如im2gps3k）包含真实经纬度
        
        # 2. 从检索结果中获取最邻近样本的经纬度作为预测结果
        # I[:, 0]表示每个测试样本的第1个近邻索引
        df['NN_idx'] = I[:, 0]  # 存储近邻索引
        # 根据近邻索引从数据库中获取预测经纬度
        df['LAT_pred'] = df['NN_idx'].apply(lambda x: database.loc[x, 'LAT'])
        df['LON_pred'] = df['NN_idx'].apply(lambda x: database.loc[x, 'LON'])
        
        # 3. 计算预测经纬度与真实经纬度的 geodesic 距离（单位：km）
        df['geodesic'] = df.apply(
            lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km, 
            axis=1
        )
        
        # 4. 保存评估结果到CSV
        save_path = f'./data/{args.dataset}_{args.index}_results_TransGeoCLIP.csv'
        df.to_csv(save_path, index=False)
        print(f"评估结果已保存至: {save_path}")
        
        # 5. 计算并打印不同距离阈值下的准确率
        total = len(df)
        print(f"总样本数: {total}")
        print(f"2500km level: {df[df['geodesic'] < 2500].shape[0] / total:.4f}")
        print(f"750km level: {df[df['geodesic'] < 750].shape[0] / total:.4f}")
        print(f"200km level: {df[df['geodesic'] < 200].shape[0] / total:.4f}")
        print(f"25km level: {df[df['geodesic'] < 25].shape[0] / total:.4f}")
        print(f"1km level: {df[df['geodesic'] < 1].shape[0] / total:.4f}")
    
if __name__ == '__main__':

    res = faiss.StandardGpuResources()

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='TransGeoCLIP')
    parser.add_argument('--dataset', type=str, default='im2gps3k')
    parser.add_argument('--database', type=str, default='mp16')
    args = parser.parse_args()
    if args.dataset == 'im2gps3k':
        args.dataset_df = pd.read_csv('./data/im2gps3k_places365.csv')
    elif args.dataset == 'yfcc4k':
        args.dataset_df = pd.read_csv('./data/yfcc4k_places365.csv')

    if args.database == 'mp16':
        args.database_df = pd.read_csv('./data/MP16_Pro_filtered.csv')

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(f''): os.makedirs(f'')
    if not os.path.exists(f'.{args.index}.index'):
        build_index(args)
    else:
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        if not os.path.exists(f'I_{args.index}_{args.dataset}.npy'):
            index = faiss.read_index(f'.{args.index}.index')
            print('read index success')
            D,I = search_index(args, index, 30) # pyright: ignore[reportGeneralTypeIssues]
            np.save(f'D_{args.index}_{args.dataset}.npy', D)
            np.save(f'I_{args.index}_{args.dataset}.npy', I)
        else:
            D = np.load(f'D_{args.index}_{args.dataset}.npy')
            I = np.load(f'I_{args.index}_{args.dataset}.npy')
        evaluate(args, I)

