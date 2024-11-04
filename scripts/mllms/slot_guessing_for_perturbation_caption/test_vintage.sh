while getopts ":m:" option; do
   case $option in
      m) # Enter model name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name

python main.py \
--eval_data_name SilentAntagonist/vintage-artworks-60k-captioned \
--eval_set_key train \
--n_eval_data_points 1000 \
--method slot-guessing-for-perturbation-caption \
--model_name $model_name \
--output_dir  \
--image_key image_url \
--caption_key short_caption