from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

import json
from collections import defaultdict
import aiofiles as aiof
import time
from datetime import datetime
import sys
sys.path.append('.')
from gpt.src.generate_from_string import continue_string, get_future_prediction
from deeppavlov import build_model, configs
import numpy as np
import pandas as pd

from aiogram.contrib.middlewares.logging import LoggingMiddleware    

#from config import TOKEN

TOKEN = open('telegram_token', 'r').read()
#PROXY = "socks5://127.0.0.1:9050"

OUTPUTFILE = "output.jsonl"

BUTTONS = {
    "SHOW_ALL": "Totall recall",
}
## Init
print('\n\nLoading model')
BERT = build_model(configs.squad.squad, download=True)

print('\n\n\n\n\n', 30*'---', '\nCreating bot')
token = open('telegram_token', 'r').read()

df = pd.read_json("data/london_2020s.jsonl", lines=True)


futures = {}

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

## End of init

def answer_if_confident(question, context, strictness=0.7):
    """
    strictness : bigger the strictness, bigger the threshold
    :return:
    """

    known_ans = "The best cars in the world are Tesla."
    context += known_ans
    known_q = "What are the best cars in the world?"
    absurd = "orange the is go happen furiously?"

    answers, _, (Eq, Ek, Ea) = BERT([context]*3, [question, known_q, absurd])

    threshold = np.sqrt(Ek*Ea)*strictness

    return answers[0] if Eq > threshold else "GPT-2 have found an answer to your question... " \
                                             "I hope you eventually find out what you are searching for."


def extracted_answers(questions, context):
    return {q: answer_if_confident(q, context) for q in questions}



def user_to_dict(user):
    return {
            "id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            }


async def write_answer(data):
    async with aiof.open(OUTPUTFILE, "a") as out:
        await out.write("{}\n".format(json.dumps(data, ensure_ascii=False)))
        await out.flush()



markup = ReplyKeyboardMarkup(resize_keyboard=True).row(
    KeyboardButton(BUTTONS["SHOW_ALL"])
)

@dp.message_handler(commands=['start'])
async def process_help_command(msg: types.Message):
    restaurant_sample = df[df.text.str.contains("\n")].sample(1)
    name = restaurant_sample["name"].values[0]
    text = restaurant_sample["text"].values[0]
    await bot.send_message(msg.from_user.id, "Hello {}, You were in {}, ".format(msg.from_user.first_name, name) +
                                      "but looks like you forgot everything... I am listening to your memories. " +
                                      "Wait for a bit and I will come back with something for you.")
    review = [t.replace("...", " ") for t in text.split("\n") if t.strip() != '']
    futures[msg.from_user.id] = get_future_prediction(review)

    await bot.send_message(
                        msg.from_user.id, 'You can ask me anything now, '
                        'or you can try to dive all by yourself in the messages from your memory.',
                        reply_markup=markup)


@dp.message_handler(regexp="Total recall.")
async def show_all(msg: types.Message):
    if msg.from_user.id in futures:
        await bot.send_message(msg.from_user.id, futures[msg.from_user.id])

@dp.message_handler()
async def echo_message(msg: types.Message):
    if msg.text in set(BUTTONS.values()):
        return
    if msg.from_user.id in futures:
        await bot.send_message(msg.from_user.id, answer_if_confident(msg.text, futures[msg.from_user.id]))
    await bot.send_message(msg.from_user.id, "Try /start")

if __name__ == '__main__':
    print('\n\n\n\n\n', 30*'---', '\nTry your bot')
    executor.start_polling(dp)