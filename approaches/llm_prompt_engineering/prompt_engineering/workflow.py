import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json

class ModelFunc:
    def __init__(self, model_path, device):
        self.device = device
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True).to(device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"加载模型或分词器时出错: {e}")

    def generate(self, prompt, user_input, max_tokens) -> str:
        try:
            with torch.no_grad():
                message = [{"role":"system","content":prompt},{"role":"user","content":user_input}]
                conversation = self.tokenizer.apply_chat_template(message,add_generation_prompt=True,tokenize=False)
                encoading = self.tokenizer(conversation,return_tensors="pt").to(self.device)
                output_ids = self.model.generate(**encoading,max_new_tokens=max_tokens,repetition_penalty=1.1,pad_token_id=self.tokenizer.eos_token_id)
                response = self.tokenizer.decode(output_ids[0])
                return response
        except Exception as e:
            raise RuntimeError(f"生成回复时出错: {e}")
        
class WorkflowExecutor:
    gpu_num=2
    if gpu_num == -1:
        device = torch.device(f"cpu")
    elif gpu_num == 'auto':
        device = torch.device(f"cuda")
    else:
        device = torch.device(f"cuda:{gpu_num}")
    
    model = ModelFunc(model_path='/data/public/models/DeepSeek-R1-Distill-Qwen-14B',device=device)

    print('device:',device)
    def answer_question(wave_length):
       
        with open(f"meta_surface_dataset_{wave_length}.json", "r") as file:
            system_prompt= json.load(file)
    
        user_input =    f"""
    Predict the optical response of the target structure with the following structural parameters:
    [100, 175, 101, 26, 236, 287, 59, 72] at the wavelength of {wave_length}nm. please think thoroughly, thoroughly, thoroughly."""
    
        response = WorkflowExecutor.model.generate(system_prompt, user_input, max_tokens=65536)
        
        return response

    def solve_problem(wave_length):
        try:
            answer = WorkflowExecutor.answer_question(wave_length)
            return answer
        except Exception as e:
            print(f"workflow failed: {str(e)}")

if __name__ == "__main__":
    wave_length = 650
    answer = WorkflowExecutor.solve_problem(wave_length=wave_length)
    # print(answer)
    with open(f"meta_surface_answer_{wave_length}_new.json", "w") as json_file:
        json.dump(answer, json_file, indent=2)
    print("complete answer")
    print("wave_length:",wave_length)
