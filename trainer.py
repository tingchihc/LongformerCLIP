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
            # TODO replace root for your image location.
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


class LongformerCLIPTrainer:
    def __init__(self,
                 vision_model="openai/clip-vit-base-patch32",
                 text_model="allenai/longformer-base-4096",
                 learning_rate=1e-5):
        """
        Trainer for Longformer CLIP model

        Args:
        vision_model (str): CLIP vision model
        text_model (str): Longformer model
        learning_rate (float): Learning rate for training
        """
        # Initialize tokenizers and processors
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model)
        self.text_tokenizer = LongformerTokenizer.from_pretrained(text_model)
        # Initialize model
        self.model = LongformerCLIPModel(
            vision_model_name=vision_model, 
            text_model_name=text_model
        )
        self.best_val_loss = float('inf')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model) # If you do not have multi-GPUs, please comment this
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
    
    def train(self, train_dataloader, val_dataloader=None, epochs=3, folder=None, save_path='/best_model.pt'):
        """
        Training loop for Longformer CLIP model
        
        Args:
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader, optional): Validation data loader
        epochs (int): Number of training epochs
        save_path (str): path to save the best model
        """

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        logger.info("start run training epoch")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    pixel_values=pixel_values
                )
                
                loss = self.model.module.compute_contrastive_loss(
                    outputs['image_embeds'], 
                    outputs['text_embeds'] 
                )
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        if val_dataloader is not None:
            val_loss = self.validate(val_dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if isinstance(self.model, torch.nn.DataParallel):
                model_to_save = self.model.module
            else:
                model_to_save = self.model
            torch.save(model_to_save.state_dict(), folder + save_path)
            self.vision_processor.save_pretrained(folder)
            self.text_tokenizer.save_pretrained(folder)
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")

    def validate(self, val_dataloader):
        """
        Validation loop for the model
        
        Args:
        val_dataloader (DataLoader): Validation data loader
        
        Returns:
        float: Average validation loss
        """
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    pixel_values=pixel_values
                )

                loss = self.model.module.compute_contrastive_loss(
                    outputs['image_embeds'], 
                    outputs['text_embeds'] 
                )
                
                total_val_loss += loss.item()
        
        return total_val_loss / len(val_dataloader)          


def main(argument):

    clip_trainer = LongformerCLIPTrainer(learning_rate=argument.learning_rate)
    
    logger.info("Preprocee Dataset with LongformerCLIP.")
    train_dataset = LongformerCLIPDataset(
        text_file=argument.input_data,
        vision_processor=clip_trainer.vision_processor,
        text_tokenizer=clip_trainer.text_tokenizer
    )
    val_dataset = LongformerCLIPDataset(
        text_file=argument.val_data,
        vision_processor=clip_trainer.vision_processor,
        text_tokenizer=clip_trainer.text_tokenizer
    )
    logger.info("Finish training/validation data preprocessing.")
    logger.info("Build up the dataloader.")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=argument.batch_size, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=argument.batch_size,
        shuffle=True
    )
    logger.info("Finish training/validation dataloader.")
    logger.info("Start training.")
    clip_trainer.train(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    epochs=argument.epochs,
                    folder=argument.save_folder
    )
    logger.info("Finish training.")
    logger.info("Save model.")
    #clip_trainer.save_model(argument.save_path)
    
if __name__ == "__main__":
    # conda activate clip-train

    """
    python trainer.py   --input_data /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/train.json \
                        --val_data /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/val.json \ 
                        --save_folder  /home/grads/tingchih2/workshop/CLIP-myself/longformerCLIP/model_record/ \
                        --batch_size 16 \
                        --epochs 3 \
                        --learning_rate 2e-5 \
    """

    parser = argparse.ArgumentParser(description="Train LongformerCLIP configurable arguments")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--val_data', type=str)
    args = parser.parse_args()
    main(args)
