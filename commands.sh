img2dataset --url_list training/laion-100k-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder training/laion-100k-data --processes_count 16 --thread_count 128 --image_size 768\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb 

img2dataset --url_list training/laion-20-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder training/laion-20-data --processes_count 1 --thread_count 4 --image_size 768\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb --number_sample_per_shard=2