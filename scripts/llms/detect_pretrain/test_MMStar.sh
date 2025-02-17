while getopts ":m:" option; do
   case $option in
      m) # Enter model name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name

python main.py \
--eval_data_name Lin-Chen/MMStar \
--eval_set_key val \
--text_key question \
--label_key answer \
--n_eval_data_points 1000 \
--method pretrain-detect \
--model_name $model_name \
--output_dir MM-Detect/scripts/llms/detect_pretrain/log/internlm2-7b \
--image_key image 