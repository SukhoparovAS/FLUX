import os
import torch
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
from PIL import Image

# 🔹 Пути к моделям
MODEL_PATH = "./models/flux-1"
DATASET_PATH = "./dataset"
TRAINED_MODEL_PATH = "./trained_model"

# 🔹 Ссылка на модель Hugging Face
MODEL_HF = "black-forest-labs/FLUX.1-schnell"


# 🔹 Функция проверки наличия модели
def is_model_downloaded():
    return os.path.exists(MODEL_PATH)


# 🔹 Функция загрузки модели (если не скачана)
def download_model():
    if not is_model_downloaded():
        print("⚡ Загружаем модель FLUX AI...")
        os.system(f"huggingface-cli download {MODEL_HF} --local-dir {MODEL_PATH}")
        print("✅ Модель загружена!")


# 🔹 Функция обучения модели на 10 фото
def train_model():
    print("🚀 Обучение модели...")

    # Используем универсальный загрузчик
    pipeline = AutoPipelineForText2Image.from_pretrained(MODEL_PATH)
    pipeline.to("cuda")

    # Загружаем изображения из папки
    if not os.path.exists(DATASET_PATH):
        print("❌ Ошибка: Папка с фото не найдена!")
        return

    images = [Image.open(os.path.join(DATASET_PATH, img)) for img in os.listdir(DATASET_PATH) if img.endswith((".jpg", ".png"))]

    if len(images) < 10:
        print("❌ Ошибка: Нужно минимум 10 фото для обучения!")
        return

    # Конфигурация LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["crossattention", "to_q", "to_v"],
        bias="none",
        task_type="TEXT_TO_IMAGE_GENERATION"
    )

    model = get_peft_model(pipeline.unet, lora_config)
    model = prepare_model_for_kbit_training(model)

    training_args = TrainingArguments(
        output_dir=TRAINED_MODEL_PATH,
        per_device_train_batch_size=1,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_dir="logs"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=images)
    trainer.train()

    model.save_pretrained(TRAINED_MODEL_PATH)
    print("✅ Обучение завершено!")


# 🔹 Функция генерации изображений
def generate_image():
    if not os.path.exists(TRAINED_MODEL_PATH):
        print("❌ Ошибка: Модель не обучена! Сначала запустите обучение.")
        return

    print("⚡ Загружаем обученную модель...")
    pipeline = AutoPipelineForText2Image.from_pretrained(TRAINED_MODEL_PATH)
    pipeline.to("cuda")

    prompt = input("Введите описание изображения: ")
    print("🎨 Генерация изображения...")
    
    image = pipeline(prompt).images[0]
    image.save("generated_image.png")
    
    print("✅ Изображение сохранено: generated_image.png")


# 🔹 Главное меню
def main():
    # Загружаем модель, если её нет
    download_model()

    print("\nВыберите действие:")
    print("1 - Обучить модель")
    print("2 - Сгенерировать изображение")
    
    choice = input("Ваш выбор: ")
    if choice == "1":
        train_model()
    elif choice == "2":
        generate_image()
    else:
        print("❌ Ошибка: Неверный ввод!")


# 🔹 Запуск программы
if __name__ == "__main__":
    main()
