import telebot
import sys
sys.path.append('.')
from gpt2.src.generate_from_string import continue_string
from deeppavlov import build_model, configs
import numpy as np

print('\n\nLoading model')
BERT = build_model(configs.squad.squad, download=True)

print('\n\n\n\n\n', 30*'---', '\nCreating bot')
token = '864065501:AAHvoUqncSS8t-_x7E1lnP7EvWJd3IM3mpM'
bot = telebot.TeleBot(token)


def answer_about_friendship(question):
    FRIEND = """Friendship is a relationship of mutual affection between people.
     Friendship is a stronger form of interpersonal bond than an association. 
     Friendship has been studied in academic fields such as communication, sociology, social psychology, anthropology, and philosophy. 
     arious academic theories of friendship have been proposed, including social exchange theory, equity theory, relational dialectics, and attachment styles.
    Although there are many forms of friendship, some of which may vary from place to place, certain characteristics are present in many types of such bonds. 
    Such characteristics include affection; kindness, love, virtue, sympathy, empathy, honesty, altruism, loyalty, generosity, forgiveness, mutual understanding and compassion, enjoyment of each other's company, trust, and the ability to be oneself, express one's feelings to others, and make mistakes without fear of judgment from the friend. Friendship is an essential aspect of relationship building skills.
    """
    return BERT([FRIEND], [question])


def answer_if_confident(question, context, strictness=1.0):
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

    return answers[0] if Eq > threshold else "Havent Found"


def extracted_answers(questions, context):
    return {q: answer_if_confident(q, context) for q in questions}


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(regexp="test")
def echo_all(message):
    bot.reply_to(message, 'Are you TESTING ME?')


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    ans = answer_about_friendship(message.text)
    print('\nQuestion:', message.text)
    print('\nAnswer:', ans[0][0])
    print('\nFull:', ans)
    continuation = continue_string(message.text, length=100)
    bot.reply_to(message, continuation)


print('Start handling')
bot.polling()