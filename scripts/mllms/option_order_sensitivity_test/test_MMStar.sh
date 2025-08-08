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
--eval_data_name Lin-Chen/MMStar \
--eval_set_key val \
--text_key question \
--n_eval_data_points 1000 \
--method option-order-sensitivity-test \
--model_name $model_name \
--output_dir outputs \
--image_key image \
$resume_flag 