import torch
import time

# 3.1 Подготовка данных
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
tensor1 = torch.rand(64, 1024, 1024)
tensor2 = torch.rand(128, 512, 512)
tensor3 = torch.rand(256, 256, 256)



# 3.2 Функция измерения времени
# Создайте функцию для измерения времени выполнения операций
def measure_time(func, cpu_val, gpu_val):
    # Используйте time.time() для измерения на CPU
    start = time.time()
    func(cpu_val)
    cpu_result = (time.time() - start) * 1000

    # Используйте torch.cuda.Event() для точного измерения на GPU
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        func(gpu_val)
        end_event.record()
        torch.cuda.synchronize()
        gpu_result = start_event.elapsed_time(end_event)
    else:
        gpu_result = None

    return cpu_result, gpu_result



# 3.3 Сравнение операций
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

funcs_dict = {
    "Матричное умножение": lambda x: torch.matmul(x, x),
    "Поэлементное сложение": lambda x: x + x,
    "Поэлементное умножение": lambda x: x * x,
    "Транспонирование": lambda x: x.transpose(0, 1),
    "Вычисление суммы всех элементов": lambda x: x.sum()
}

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
def make_operations(tensor):
    print(f"{'Операция':35} | {'CPU (мс)':10} | {'GPU (мс)':10} | {'Ускорение':10}")
    for name, func in funcs_dict.items():
        cpu_val = tensor.to(device_cpu)
        gpu_val = tensor.to(device_gpu)
        work_time = measure_time(func, cpu_val, gpu_val)
        cpu_time = work_time[0]
        gpu_time = work_time[1]
        speedup = cpu_time/gpu_time if gpu_time is not None else None
        print(f"{name:35} | {cpu_time:10.1f} | {gpu_time or 0:10.1f} | {speedup or ' ':10}")

make_operations(tensor1)
print()
make_operations(tensor2)
print()
make_operations(tensor3)