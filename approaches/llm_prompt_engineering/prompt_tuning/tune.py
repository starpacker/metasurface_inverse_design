from peft import PromptTuningConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers import BeamSearchScorer
import torch
import torch.nn.functional as F
import json
import os
import re
from typing import Dict, List
import yaml
from ipdb import set_trace
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

class TunedModel:
    def __init__(self, model_path: str, device: torch.device):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_num'])
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch.cuda.set_device(self.device)
        print('当前使用设备:', self.device)
        print('实际物理GPU:', torch.cuda.get_device_name(self.device))

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # device_map=device,
            trust_remote_code=True
        ).to(device)
        self.peft_config = None
        self.model = None

    def prepare_for_tuning(self, system_prompt: str,response_format: str, num_virtual_tokens: int = 20):
        """
        Prepare the model for tuning with a system prompt .

        Args:
            system_prompt (str): The initial prompt text for tuning.
            num_virtual_tokens (int, optional): Number of virtual tokens for prompt tuning. Defaults to 20.
        """
        self.peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=system_prompt + response_format,
            num_virtual_tokens=num_virtual_tokens,
            token_dim=self.base_model.config.hidden_size,
            tokenizer_name_or_path=self.tokenizer.name_or_path
        )
        self.model = get_peft_model(self.base_model, self.peft_config)

    def generate(self, user_input: str, max_tokens: int = 16384) -> str:
        try:
            with torch.no_grad():
                messages = [{"role": "user", "content": user_input}]
                conversation = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self.tokenizer(
                    conversation,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"生成失败: {str(e)}")

class OpticalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: List[dict], tokenizer: AutoTokenizer, device):
        self.data = dataset
        self.tokenizer = tokenizer
        self.device = device


    def __len__(self):
        return len(self.data)   

    def __getitem__(self, idx) -> Dict:
        sample = self.data[idx]
        # text = f"结构参数: {sample['structure']} @ {sample['wavelength']}nm → 响应: {sample['response']}" 
        text = (f"structure: {sample['structure']}\n"
                f"spectrum_response: sample['spectrum_response']"
                ) 
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        return {
            'input_ids': torch.tensor(encoding['input_ids']).to(self.device),
            'attention_mask': torch.tensor(encoding['attention_mask']).to(self.device),
            'labels': torch.tensor(encoding['input_ids']).to(self.device)
        }

class DataLoader:
    @staticmethod
    def _validate_data(data: List[Dict]):
        """数据完整性检查"""
        # required_keys = ['structure', 'wavelength', 'response']
        required_keys = ['structure','spectrum_response']
        for sample in data:
            missing = [k for k in required_keys if k not in sample]
            if missing:
                raise ValueError(f"数据样本缺少字段: {missing}")

                
    @staticmethod
    def load_dataset(wave_length: int, tokenizer, device) -> OpticalDataset:
        """加载训练数据文件"""
        data_path = f"/data/group_003/prompt_tuning/meta_surface_dataset_{wave_length}.json"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
            DataLoader._validate_data(data)
        return OpticalDataset(data, tokenizer,device)

    @staticmethod
    def load_prompt_config(wave_length: int) -> Dict:
        """加载提示词配置文件"""
        prompt_path = f"/data/group_003/prompt_tuning/prompt_config_{wave_length}.yaml"  # 使用YAML格式更易管理
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"提示词配置不存在: {prompt_path}")

        with open(prompt_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 配置验证
        required_keys = ['system_prompt', 'response_format']    #重新生成yaml文件包含这两种格式
        if not all(k in config for k in required_keys):
            missing = [k for k in required_keys if k not in config]
            raise ValueError(f"提示词配置缺少必要字段: {missing}")
            
        return config



class WorkflowExecutor:
    def __init__(self, config: Dict):
        self.device = self._get_device(config['gpu_num'])
        self.model = TunedModel(config['model_path'], self.device)
        self.tokenizer = self.model.tokenizer

        print('device:',self.device)
        
        # 物理参数约束
        self.physical_constraints = config.get('physics_constraints', {})

    @staticmethod
    def _get_device(gpu_num: int) -> torch.device:
        # if gpu_num == -1:
        #     return torch.device(f"cpu")
        # elif gpu_num == 'auto':
        #     return torch.device(f"cuda")
        # else:
        #     return torch.device(f"cuda:{gpu_num}")
        if gpu_num == -1:
            return torch.device("cpu")
        return torch.device(f"cuda:{gpu_num}")  # 这里固定为0，因为CUDA_VISIBLE_DEVICES已限定物理设备

    def _extract_values(self, text: str) -> Dict:

        # 尝试从文本中提取 JSON 部分
        json_pattern = r'```json\n(.*?)\n```'
        json_match = re.search(json_pattern, text, re.DOTALL)
        if not json_match:
            print("警告: 未找到 JSON 块")
            return {}

        try:
            # 解析 JSON
            data = json.loads(json_match.group(1))
            spectrum_response = data.get('spectrum_response', {})

            # 提取复数参数
            result = {
                'r_rl_real': spectrum_response.get('r_rl', {}).get('real'),
                'r_rl_imag': spectrum_response.get('r_rl', {}).get('imag'),
                'r_lr_real': spectrum_response.get('r_lr', {}).get('real'),
                'r_lr_imag': spectrum_response.get('r_lr', {}).get('imag'),
                'r_rr_real': spectrum_response.get('r_rr', {}).get('real'),
                'r_rr_imag': spectrum_response.get('r_rr', {}).get('imag')
            }

            # 验证复数对完整性
            for prefix in ['r_rl', 'r_lr', 'r_rr']:
                real = result.get(f'{prefix}_real')
                imag = result.get(f'{prefix}_imag')
                if (real is not None) ^ (imag is not None):  # 异或判断
                    print(f"警告: {prefix} 的实部/虚部不完整")
            # 返回非空值
            return {k: v for k, v in result.items() if v is not None}

        except json.JSONDecodeError:
            print("错误: JSON 解析失败")
            return {}
        
    def _physics_loss(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        
        for param in ['r_rl', 'r_lr', 'r_rr']:
            # 安全获取并类型检查
            pred_real = predictions.get(f'{param}_real')
            pred_imag = predictions.get(f'{param}_imag')
            tgt_real = targets[f'{param}_real']
            tgt_imag = targets[f'{param}_imag']
            
            # 统一检查所有必要参数
            if None in (pred_real, pred_imag, tgt_real, tgt_imag):
                continue
                
            try:
                # 转换并确保设备一致
                pred = torch.tensor([pred_real, pred_imag], 
                                dtype=torch.float32,
                                device=self.device)
                tgt = torch.tensor([tgt_real, tgt_imag],
                                dtype=torch.float32,
                                device=self.device)
                
                # 计算复数MSE
                loss = F.mse_loss(pred, tgt)
                total_loss += loss
            except (TypeError, RuntimeError) as e:
                print(f"参数 {param} 计算异常: {str(e)}")
                continue

            return total_loss 
        
        return torch.tensor(10.0, device=self.device)

    def train(self, wave_length: int, train_config: Dict):
        # 显存状态监控
        print(f"可用显存: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")
        try:
            # 1. 模块化加载
            dataset = DataLoader.load_dataset(wave_length,self.tokenizer,self.device)
            prompt_config = DataLoader.load_prompt_config(wave_length)

                        # 验证设备一致性
            print("\n=== 设备配置验证 ===")
            print(f"Torch可见设备数: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"模型参数设备: {next(self.model.base_model.parameters()).device}")
            
            # 验证数据设备
            sample = next(iter(dataset))
            print(f"数据样本设备: {sample['input_ids'].device if isinstance(sample['input_ids'], torch.Tensor) else 'CPUs'}")

            
            # 2. 提示词初始化
            self.model.prepare_for_tuning(
                system_prompt=prompt_config['system_prompt'],
                response_format=prompt_config['response_format']
            )
            # 3. 配置训练参数
            training_args = TrainingArguments(
                output_dir="./prompt_checkpoints",
                learning_rate=train_config.get('lr', 3e-4),
                per_device_train_batch_size=train_config.get('batch_size', 2),
                num_train_epochs=train_config.get('epochs', 10),
                logging_dir='./logs',
                report_to=['tensorboard'],
                save_strategy="epoch"
            )

            # 4. 自定义训练器
            class PhysicsAwareTrainer(Trainer):
                def __init__(self, workflow_executor=None, **kwargs):
                    super().__init__(**kwargs)
                    self.workflow_executor = workflow_executor  # 保存外部Executor实例
                
                
                def compute_loss(self, model, inputs, return_outputs=False):
                    
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # 从当前batch的logits解码预测文本
                    pred_texts = self.workflow_executor.tokenizer.batch_decode(
                        torch.argmax(logits, dim=-1),
                        skip_special_tokens=True
                    )

                    
                    set_trace()
                    
                    predictions = [self.workflow_executor._extract_values(t) for t in pred_texts]
                    physics_loss = self.workflow_executor._physics_loss(predictions, inputs["labels"])
                    
                    total_loss = 1 * physics_loss
                    print("physics_loss:",physics_loss)
                    return (total_loss, outputs) if return_outputs else total_loss
            
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            data_collator.device = self.device
            # 5. 执行训练
            trainer = PhysicsAwareTrainer(
                workflow_executor=self,
                model=self.model.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator
            )
            trainer.train()
            
            # 6. 保存适配器
            # self.model.model.save_pretrained(f"./tuned_prompts/wl_{wave_length}")

        except Exception as e:
            raise RuntimeError(f"训练流程失败: {str(e)}")

    def predict(self, wave_length: int, params: List[int]) -> Dict:
        try:
            # 加载适配器
            # self.model.model.load_adapter(f"./tuned_prompts/wl_{wave_length}")
            
            user_input = f"""
            Predict the optical response of the target structure with parameters:
            {params} at {wave_length}nm. Provide detailed physical analysis.
            """
            
            response = self.model.generate(user_input)
            return {
                "text": response,
                "metrics": self._extract_values(response)
            }
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")

if __name__ == "__main__":
    config = {
        "model_path": "/data/public/models/DeepSeek-R1-Distill-Qwen-14B",
        "gpu_num": 0,
        "physics_constraints": {
            "conservation_law": True,
            "valid_ranges": {
                "reflection": [-1,1],
            }
        }
    }
    
    workflow = WorkflowExecutor(config)
    
    # 训练流程
    workflow.train(
        wave_length=650,
        train_config={
            "lr": 3e-4,
            "batch_size": 1,
            "epochs": 100
        }
    )
    
    # 推理流程
    result = workflow.predict(
        wave_length=650,
        params=[100, 175, 101, 26, 236, 287, 59, 72]
    )
    
    print("预测结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
"""
idx= 150
tensor(650.)
tensor([ 0.0637, -0.1679,  0.1198, -0.0592,  0.7264,  0.1704])
pra [100, 175, 101, 26, 236, 287, 59, 72]
"""
