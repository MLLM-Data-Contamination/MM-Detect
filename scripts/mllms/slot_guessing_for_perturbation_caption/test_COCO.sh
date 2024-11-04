while getopts ":m:" option; do
   case $option in
      m) # Enter model name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name

python main.py \
--eval_data_name lmms-lab/COCO-Caption \
--eval_set_key val \
--n_eval_data_points 1000 \
--method slot-guessing-for-perturbation-caption \
--model_name $model_name \
--output_dir  \
--image_key image \
--caption_key answer