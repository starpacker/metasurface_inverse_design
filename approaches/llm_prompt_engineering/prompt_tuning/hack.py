import torch
import time

def occupy_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return

    device = torch.device("cuda:3")
    
    try:
        # 获取GPU总显存
        total_mem = torch.cuda.get_device_properties(device).total_memory
        

        allocate_mem = total_mem - 1024**3 * 2 
        
        # 计算可以存储的float32元素数量（每个元素4字节）
        num_elements = allocate_mem // 4
        
        # 创建未初始化的张量（最快的内存分配方式）
        dummy_tensor = torch.empty(
            num_elements, 
            dtype=torch.float32,
            device=device
        )
        
        print(f"已成功分配 {allocate_mem/1024**3:.2f} GB 显存")
        print("程序将持续运行占据显存，按Ctrl+C退出...")
        
        # 保持程序持续运行
        while True:
            time.sleep(360)  # 每小时唤醒一次避免被系统挂起
            
    except KeyboardInterrupt:
        print("\n释放显存...")
        del dummy_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    occupy_gpu_memory()
