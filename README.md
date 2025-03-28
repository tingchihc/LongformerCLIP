# LongformerCLIP

## Dataset
To prepare your own dataset for running Longformer.
- You should resize your images to 214*214.
- Your dataset json template can follow this:
    ```
    {
        "0": {"product": "text", "image": "image_location"}, ...
    }
    ```

## Train & Inference
```
    python trainer.py   --input_data /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/train.json \
                        --val_data /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/val.json \ 
                        --save_folder  /home/grads/tingchih2/workshop/CLIP-myself/longformerCLIP/model_record/ \
                        --batch_size 16 \
                        --epochs 3 \
                        --learning_rate 2e-5 \
```

```
    python tester.py    --model_path /home/grads/tingchih2/workshop/CLIP-myself/longformerCLIP/model_record \ 
                        --test_data  /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/val.json \
                        --batch_size 16  \
                        --img_txt_similarity_path /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/87737.jpg \
                        --img_txt_similarity_text 牛蒡茶 \
                        --zero_shot_img /home/grads/tingchih2/workshop/CLIP-myself/my_dataset/images/87737.jpg

```

## Acknowledgement
[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)  
[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)  
