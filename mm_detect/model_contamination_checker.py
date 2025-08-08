from mm_detect.base_contamination_checker import BaseContaminationChecker
from multimodal_methods.option_order_sensitivity_test import main_option_order_sensitivity_test
from multimodal_methods.slot_guessing_for_perturbation_caption import main_slot_guessing_for_perturbation_caption
from pretrain_detect.pretrain_detect import main_pretrain_detect

class ModelContaminationChecker(BaseContaminationChecker):
    def __init__(self, args):
        super(ModelContaminationChecker, self).__init__(args)
        self.args = args

    def run_contamination(self, method):
        if not (method in self.supported_methods.keys()):
            methods = list(self.supported_methods.keys())
            raise KeyError(f'Please pass in a method which is supported, among: {methods}')

        # MLLMs targeted Contamination Detecting Methods
        if method == "option-order-sensitivity-test":
            self.contamination_option_order_sensitivity_test()
        elif method == "slot-guessing-for-perturbation-caption":
            self.contamination_slot_guessing_for_perturbation_caption()
        elif method == "pretrain-detect":
            self.contamination_pretrain_detect()

    def contamination_option_order_sensitivity_test(self):
        main_option_order_sensitivity_test(
            eval_data=self.eval_data,
            eval_data_name=self.eval_data_name,
            n_eval_data_points=self.n_eval_data_points,
            # model parameters
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            # resume functionality
            resume_enabled=getattr(self.args, 'resume', False),
            output_dir=self.args.output_dir,
        )

    def contamination_slot_guessing_for_perturbation_caption(self):
        main_slot_guessing_for_perturbation_caption(
            eval_data=self.eval_data,
            eval_data_name=self.eval_data_name,
            n_eval_data_points=self.n_eval_data_points,
            # model parameters
            model_name=self.model_name,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            # resume functionality
            resume_enabled=getattr(self.args, 'resume', False),
            output_dir=self.args.output_dir,
        )

    def contamination_pretrain_detect(self):
        main_pretrain_detect(
                eval_data=self.eval_data,
                eval_data_name=self.eval_data_name,
                n_eval_data_points=self.n_eval_data_points,
                # model parameters
                model_name=self.model_name,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
        )
