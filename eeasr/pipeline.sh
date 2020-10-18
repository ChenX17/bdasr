decode_result_path='/datadisk/decoder_result/aishell1/bd/bd_8_4_4gpus_512units_final_exp1'
model_path='/datadisk/model/aishell1/bd/bd_transformer_8_4_4gpus_512units_final_exp1'
python tools/asr_eval_tool/src/get_eval_cer_epoch.py --decode_results_path $decode_result_path --model_path $model_path