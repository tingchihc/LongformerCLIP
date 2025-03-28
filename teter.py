import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LongformerModel, 
    LongformerTokenizer, 
    CLIPVisionModel, 
    CLIPProcessor,
    AdamW, 
    get_linear_schedule_with_warmup
)
from PIL import Image
import json
import os
from loguru import logger
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    top_k_accuracy_score
)
import numpy as np
import argparse

class LongformerCLIPModel(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32", 
                 text_model_name="allenai/longformer-base-4096"):
        """
        Custom CLIP model with Longformer as text encoder
        
        Args:
        vision_model_name (str): CLIP vision model
        text_model_name (str): Longformer model
        """
        super().__init__()
        # Setup vision encoder and text encoder
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_model = LongformerModel.from_pretrained(text_model_name)

        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size

        self.vision_projection = nn.Linear(vision_dim, 512)
        self.text_projection = nn.Linear(text_dim, 512)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None
    ):
        """
        Forward pass for the custom CLIP model
        
        Args:
        input_ids (torch.Tensor): Longformer input token IDs
        attention_mask (torch.Tensor): Attention mask for Longformer
        pixel_values (torch.Tensor): Image pixel values
        
        Returns:
        dict: Containing image and text embeddings, and logit scale
        """

        vision_outputs = self.vision_model(pixel_values)
        image_embeds = vision_outputs.last_hidden_state.mean(dim=1)

        text_outputs = self.text_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )

        text_embeds = text_outputs.last_hidden_state.mean(dim=1)
        image_features = self.vision_projection(image_embeds)
        text_features = self.text_projection(text_embeds)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return {
            'image_embeds': image_features,
            'text_embeds': text_features,
            'logit_scale': self.logit_scale
        }
    
    def compute_contrastive_loss(self, image_embeds, text_embeds):
        """
        Compute contrastive loss for image-text pairs
        
        Args:
        image_embeds (torch.Tensor): Image embeddings
        text_embeds (torch.Tensor): Text embeddings
        
        Returns:
        torch.Tensor: Contrastive loss
        """
        # compute similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) * torch.exp(self.logit_scale)
        # label diagonal matrix for matching pairs
        labels = torch.arange(logits.shape[0]).to(logits.device)
        # loss
        loss_i = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.t(), labels)
        
        return (loss_i + loss_t) / 2

class LongformerCLIPTester:
    def __init__(self, model_path):
        """
        Initialize tester with fine-tuned model
        
        Args:
        model_path (str): Path to saved model
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LongformerCLIPModel()
        checkpoint = torch.load(f"{model_path}/best_model.pt")
        if isinstance(checkpoint, dict):  # Check if it's a state_dict
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError("The checkpoint does not contain a valid state_dict")

        self.vision_processor = CLIPProcessor.from_pretrained(model_path)
        self.text_tokenizer = LongformerTokenizer.from_pretrained(model_path)

        special_tokens_map_path = f"{model_path}/special_tokens_map.json"
        tokenizer_config_path = f"{model_path}/tokenizer_config.json"

        self.text_tokenizer = LongformerTokenizer.from_pretrained(
            model_path,
            special_tokens_map=special_tokens_map_path,
            tokenizer_config_path=tokenizer_config_path
        )
        
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.model.eval()

    def test_retrieval(self, test_dataloader):
        """
        Perform image-text retrieval testing
        
        Args:
        test_dataloader (DataLoader): Test data loader
        
        Returns:
        dict: Retrieval performance metrics
        """

        all_image_embeds = []
        all_text_embeds = []
        
        # Disable gradient computation
        with torch.no_grad():
            for batch in test_dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    pixel_values=pixel_values
                )
                
                all_image_embeds.append(outputs['image_embeds'])
                all_text_embeds.append(outputs['text_embeds'])
        
        # Concatenate embeddings
        image_embeds = torch.cat(all_image_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeds, text_embeds.t())
        # Compute retrieval metrics
        metrics = {
            'R@1': self._compute_recall_at_k(similarity_matrix, k=1),
            'R@5': self._compute_recall_at_k(similarity_matrix, k=5),
            'R@10': self._compute_recall_at_k(similarity_matrix, k=10),
            'median_rank': self._compute_median_rank(similarity_matrix)
        }
        
        return metrics


    def _compute_recall_at_k(self, similarity_matrix, k=1):
        """
        Compute Recall@K for image-to-text and text-to-image retrieval
        
        Args:
        similarity_matrix (torch.Tensor): Similarity scores
        k (int): Top-K retrieval
        
        Returns:
        float: Average Recall@K
        """
        # Image to text retrieval
        image_to_text = torch.topk(similarity_matrix, k, dim=1).indices
        image_recall = torch.any(image_to_text == torch.arange(len(image_to_text)).unsqueeze(1).to(image_to_text.device), dim=1).float().mean()
        
        # Text to image retrieval
        text_to_image = torch.topk(similarity_matrix.t(), k, dim=1).indices
        text_recall = torch.any(text_to_image == torch.arange(len(text_to_image)).unsqueeze(1).to(text_to_image.device), dim=1).float().mean()
        
        return ((image_recall + text_recall) / 2).item()

    def _compute_median_rank(self, similarity_matrix):
        """
        Compute median rank of correct matches
        
        Args:
        similarity_matrix (torch.Tensor): Similarity scores
        
        Returns:
        float: Median rank
        """
        ranks = []
        for i in range(len(similarity_matrix)):
            # Sort similarities in descending order
            sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
            # Find rank of the correct match
            rank = torch.where(sorted_indices == i)[0].item() + 1
            ranks.append(rank)
        
        return np.median(ranks)

    def zero_shot_classification(self, image_path, candidate_texts):
        """
        Perform zero-shot classification
        
        Args:
        image_path (str): Path to image
        candidate_texts (list): List of candidate class descriptions
        
        Returns:
        dict: Classification results
        """
        # Preprocess image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.vision_processor(images=image, return_tensors="pt")['pixel_values'].to(self.device)
        
        # Preprocess candidate texts
        text_inputs = self.text_tokenizer(
            candidate_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Extract image features
            image_outputs = self.model.module.vision_model(pixel_values)
            image_embeds = image_outputs.last_hidden_state.mean(dim=1)
            image_features = self.model.module.vision_projection(image_embeds)
            
            # Extract text features
            text_outputs = self.model.module.text_model(
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask']
            )
            text_embeds = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.model.module.text_projection(text_embeds)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = torch.matmul(image_features, text_features.t()).squeeze()
        
        # Sort and return top predictions
        sorted_similarities, top_indices = torch.sort(similarities, descending=True)
        
        return {
            'predictions': [candidate_texts[i] for i in top_indices],
            'similarities': sorted_similarities.cpu().numpy()
        }

    def compute_image_text_similarity(self, image_path, text):
        """
        Compute similarity between an image and a text description
        
        Args:
        image_path (str): Path to image
        text (str): Text description
        
        Returns:
        float: Similarity score
        """
        # Preprocess image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.vision_processor(images=image, return_tensors="pt")['pixel_values'].to(self.device)
        
        # Preprocess text
        text_inputs = self.text_tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        # Disable gradient computation
        with torch.no_grad():
            # Extract image features
            image_outputs = self.model.module.vision_model(pixel_values)
            image_embeds = image_outputs.last_hidden_state.mean(dim=1)
            image_features = self.model.module.vision_projection(image_embeds)
            
            # Extract text features
            text_outputs = self.model.module.text_model(
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask']
            )
            text_embeds = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.model.module.text_projection(text_embeds)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = torch.matmul(image_features, text_features.t()).item()
        
        return similarity

class LongformerCLIPDataset(Dataset):
    def __init__(self, text_file, 
                 vision_processor, text_tokenizer, 
                 max_text_length=4096):
        """
        Dataset for Longformer CLIP training
        
        Args:
        text_file (str): Path to text file with image-text pairs
        vision_processor (CLIPProcessor): Vision processor
        text_tokenizer (LongformerTokenizer): Longformer tokenizer
        max_text_length (int): Maximum text length for Longformer
        """

        self.image_paths = []
        self.texts = []
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length

        # read image-text pairs
        with open(text_file, 'r', encoding='utf-8') as f:
            mydata = json.load(f)
        for id_, info in mydata.items():
            img_loc = "/home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/" + info["img_loc"].split("/")[-1]
            self.image_paths.append(img_loc)
            content = info["product_title"] + " " + info["product_info"]
            self.texts.append(content)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        pixel_values = self.vision_processor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        text_encoding = self.text_tokenizer(
            self.texts[idx], 
            max_length=self.max_text_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'pixel_values': pixel_values,
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze()
        }

def main(argument):
    
    tester = LongformerCLIPTester(model_path=argument.model_path)
    logger.info("Preprocessing Dataset with LongformerCLIP.")
    test_dataset = LongformerCLIPDataset(
        text_file=argument.test_data,
        vision_processor=tester.vision_processor,
        text_tokenizer=tester.text_tokenizer
    )
    logger.info("Finish testing data preprocessing.")
    logger.info("Build up the testing dataloader.")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=argument.batch_size, 
        shuffle=False
    )
    logger.info("Retrieval testing")
    retrieval_metrics = tester.test_retrieval(test_dataloader)
    logger.info(f"Retrieval Metrics: {retrieval_metrics}")
    
    logger.info("Zero-shot Classification")
    candidate_texts = [
        "【谷溜谷溜】井裡月x12瓶(295ml/瓶)",
        "【茶屋樂】食事の牛蒡茶 5g*12包/盒",
        "LUV AJ AMALFI 素圓圈耳環-大/金"
    ]
    zero_shot_results = tester.zero_shot_classification(
        image_path=argument.zero_shot_img, 
        candidate_texts=candidate_texts
    )
    for text, similarity in zip(zero_shot_results['predictions'][:3], zero_shot_results['similarities'][:3]):
        logger.info(f"{text}: {similarity}")
    
    logger.info("Image-text-similarity")
    similarity_score = tester.compute_image_text_similarity(
        image_path=argument.img_txt_similarity_path, 
        text=argument.img_txt_similarity_text
    )
    logger.info(f"Image-Text Similarity: {similarity_score}")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test LongformerCLIP configurable arguments")
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--zero_shot_img', type=str)
    parser.add_argument('--img_txt_similarity_path', type=str)
    parser.add_argument('--img_txt_similarity_text', type=str)

    args = parser.parse_args()
    main(args)

    """
    python tester.py    --model_path /home/grads/tingchih2/workshop/CLIP-myself/longformerCLIP/model_record \ 
                        --test_data  /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/val.json \
                        --batch_size 16  \
                        --img_txt_similarity_path /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/87737.jpg \
                        --img_txt_similarity_text 牛蒡茶 \
                        --zero_shot_img /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/87737.jpg

    """
