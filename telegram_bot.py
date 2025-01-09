import logging
import asyncio
import nest_asyncio
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import openpyxl
from dict_extractor.dict_extractor import DictionaryTermExtractor
from test_model import extract_terms_with_model

# Применяем nest_asyncio для работы с активным циклом событий
nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Токен Telegram-бота
TELEGRAM_TOKEN = "8165832655:AAEHo5fVYEGoHnB_FDR1LB_uYhekQHvlc_k"

TERMS_DIR = "terminator/terms_extractor/dict_extractor/ngramm_lemma_terms"

# Инициализация DictionaryTermExtractor
dict_extractor = DictionaryTermExtractor(terms_dir=TERMS_DIR)

# Временное хранилище терминов для экспорта
TEMP_TERMS = []

def extract_terms_from_text(text):
    try:
        # Извлечение терминов из словаря
        terms_dict = set(dict_extractor.extract_terms(text))
        # Извлечение терминов из модели
        terms_model = set(extract_terms_with_model(text))

        # Фильтруем результаты модели, оставляя только те, которые есть в словаре
        filtered_model_terms = terms_model & terms_dict

        # Итоговый список терминов
        combined_terms = list(terms_dict | filtered_model_terms)
        return combined_terms
    except Exception as e:
        logging.error(f"Ошибка извлечения терминов: {e}", exc_info=True)
        return []


# Создание файла Excel
def create_excel_file(terms, filename="terms.xlsx"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Термины"

    ws.append(["Термин"])  # Заголовок

    for term in terms:
        ws.append([term])

    wb.save(filename)
    return filename

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик команды /start. Приветствует пользователя и объясняет функционал бота.
    """
    welcome_message = (
        "Здравствуйте! Я ваш ассистент по анализу текста и выделению терминов.\n\n"
        "🛠 Возможности:\n"
        "1️⃣ Выделение терминов с использованием словаря и обученной модели.\n"
        "2️⃣ Экспорт найденных терминов в файл Excel.\n\n"
        "📋 Отправьте текст или файл, чтобы начать анализ.\n"
        
    )
    await update.message.reply_text(welcome_message)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик текстовых сообщений. Выделяет термины из текста и возвращает результаты.
    """
    global TEMP_TERMS
    user_text = update.message.text
    TEMP_TERMS = extract_terms_from_text(user_text)

    if TEMP_TERMS:
        keyboard = [
            [InlineKeyboardButton("📂 Экспорт в Excel", callback_data="export_excel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        response = (
            "📑 **Результаты анализа:**\n\n"
            "Обнаруженные термины:\n" + ", ".join(TEMP_TERMS) +
            "\n\nВы можете экспортировать результаты в файл Excel, нажав на кнопку ниже."
        )
        await update.message.reply_text(response, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "⚠️ Термины не найдены. Проверьте текст и попробуйте снова.",
            parse_mode="Markdown"
        )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик файлов. Извлекает текст из файла и выделяет термины.
    """
    global TEMP_TERMS
    document = update.message.document
    file_path = await document.get_file()
    file_name = f"temp_{document.file_name}"
    await file_path.download_to_drive(file_name)

    try:
        # Чтение содержимого файла
        logging.info(f"Файл получен: {file_name}")
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()

        TEMP_TERMS = extract_terms_from_text(text)

        if TEMP_TERMS:
            keyboard = [
                [InlineKeyboardButton("📂 Экспорт в Excel", callback_data="export_excel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            response = (
                "📑 **Результаты анализа файла:**\n\n"
                "Обнаруженные термины:\n" + ", ".join(TEMP_TERMS) +
                "\n\nВы можете экспортировать результаты в файл Excel, нажав на кнопку ниже."
            )
            await update.message.reply_text(response, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(
                "⚠️ Термины не найдены в файле. Проверьте содержимое и повторите попытку.",
                parse_mode="Markdown"
            )
    except Exception as e:
        logging.error(f"Ошибка обработки файла: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Не удалось обработать файл. Убедитесь, что это текстовый файл.",
            parse_mode="Markdown"
        )
    finally:
        # Удаляем временный файл
        if os.path.exists(file_name):
            os.remove(file_name)


# Обработчик кнопки "Экспорт в Excel"
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "export_excel":
        filename = create_excel_file(TEMP_TERMS)
        await query.message.reply_document(document=open(filename, "rb"))
        os.remove(filename)

# Основная функция
async def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    # Запуск бота
    await application.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
