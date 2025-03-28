# LongformerCLIP
This repo is to replace text encoder into Longformer.

"""
    python trainer.py   --input_data /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/train.json \
                        --val_data /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/val.json \ 
                        --save_folder  /home/grads/tingchih2/workshop/CLIP-myself/longformerCLIP/model_record/ \
                        --batch_size 16 \
                        --epochs 3 \
                        --learning_rate 2e-5 \
"""

"""
    python tester.py    --model_path /home/grads/tingchih2/workshop/CLIP-myself/longformerCLIP/model_record \ 
                        --test_data  /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/val.json \
                        --batch_size 16  \
                        --img_txt_similarity_path /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/87737.jpg \
                        --img_txt_similarity_text 牛蒡茶 \
                        --zero_shot_img /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/87737.jpg

"""
