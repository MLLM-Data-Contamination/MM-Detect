while getopts ":m:" option; do
   case $option in
      m) # Enter model name
         model_name=$OPTARG;;
   esac
done

echo "model name ", $model_name

python main.py \
--eval_data_name derek-thomas/ScienceQA \
--eval_set_key test \
--text_key question \
--n_eval_data_points 1340  \
--method option-order-sensitivity-test \
--model_name $model_name \
--output_dir  \
--image_key image 