resume_flag=""

while getopts ":m:r" option; do
   case $option in
      m) # Enter model name
         model_name=$OPTARG;;
      r) # Resume from checkpoint
         resume_flag="--resume";;
   esac
done

echo "model name ", $model_name

python main.py \
--eval_data_name lmms-lab/NoCaps \
--eval_set_key validation \
--n_eval_data_points 1000 \
--method slot-guessing-for-perturbation-caption \
--model_name $model_name \
--output_dir outputs \
--image_key image \
--caption_key annotations_captions \
$resume_flag