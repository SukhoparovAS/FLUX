from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
import torch
import os
from PIL import Image

def train_model():
    print("Обучение модели...")
    
    # Загружаем основную модель
    pipeline = StableDiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
    pipeline.to("cuda")

    # Загружаем изображения
    dataset_path = "dataset"
    images = [Image.open(os.path.join(dataset_path, img)) for img in os.listdir(dataset_path)]

    # Подготовка LoRA-конфигурации
    lora_config = LoraConfig(
        r=8,  # Размер адаптера
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["crossattention", "to_q", "to_v"],  # Обучаемые слои
        bias="none",
        task_type="TEXT_TO_IMAGE_GENERATION"
    )

    # Добавляем LoRA в модель
    model = get_peft_model(pipeline.unet, lora_config)
    model = prepare_model_for_kbit_training(model)

    # Настройки обучения
    training_args = TrainingArguments(
        output_dir="trained_model",
        per_device_train_batch_size=1,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_dir="logs"
    )

    # Запускаем обучение
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=images
    )

    trainer.train()

    # Сохраняем модель
    model.save_pretrained("trained_model")
    print("Обучение завершено!")

# Функция генерации изображения
def generate_image():
    if not os.path.exists("trained_model"):
        print("Ошибка: модель не обучена!")
        return

    pipeline = StableDiffusionPipeline.from_pretrained("trained_model")
    pipeline.to("cuda")

    prompt = input("Введите промпт для генерации: ")
    image = pipeline(prompt).images[0]
    image.save("generated_image.png")
    print("Изображение сохранено как generated_image.png")

# Главное меню
def main():
    print("Выберите действие:")
    print("1 - Обучить модель")
    print("2 - Сгенерировать изображение")
    
    choice = input("Ваш выбор: ")
    if choice == "1":
        train_model()
    elif choice == "2":
        generate_image()
    else:
        print("Ошибка: неверный ввод!")

if __name__ == "__main__":
    main()
