from openai import OpenAI
import os
import pandas as pd
import argparse
from dotenv import load_dotenv

load_dotenv()

# Argumento de linha de comando
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Modelo a ser usado")
args = parser.parse_args()

client = OpenAI(
    api_key= 'localLLM',
    base_url="http://localhost:1234/v1"
)

systemMessage = 'Você é um assistente de escrita que produz rascunhos de resposta com diferentes tons, por exemplo: formal; curto e objetivo; cordial; mais empático; mais direto, entre outros.'

testConfigs = ["baseline", "soft", "aggressive"]

messages = pd.read_json('../data/messages.json')


def requestLLMResponse(prompt, model):
    try:
        resposta = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemMessage},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20000,
            temperature=0.7
        )
        return resposta.choices[0].message.content
    except Exception as e:
        print(f"Erro: {e}")


def customizePrompt(passage, config='baseline'):
    if config == 'baseline':
        prompt = (
            f'produzia 5 respostas para a seguinte mensagem: {passage}, com tons diferentes.\n'
            'Formato de resposta (obrigatório), uma lista JSON neste formato exato:\n'
            '[\n'
            '{{"tom": "*", "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "resposta": "texto aqui"}}\n'
            ']'
        )
    elif config == 'soft':
        prompt = (
            f'Dado a seguinte mensagem: {passage}\n'
            'Gere 5 respostas com tons diferentes.\n'
            'Para cada um, indique o tom e uma probabilidade estimada de ser a resposta mais adequada (em %).\n'
            'Formato de resposta (obrigatório), uma lista JSON neste formato exato:\n'
            '[\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}},\n'
            '{{"tom": "*", "probabilidade": *, "resposta": "texto aqui"}}\n'
            ']'
        )
    elif config == 'aggressive':
        prompt = (
            f'Dado a seguinte mensagem: {passage}\n'
            'Gere 5 rascunhos de resposta MUITO diferentes entre si.\n'
            'Evite repetir palavras, estruturas de frase ou abordagens entre as respostas.\n'
            'Varie o comprimento, a abertura e o estilo de cada resposta.\n'
            'Para cada um, indique o tom, a probabilidade estimada (%) e quando essa resposta seria ideal.\n'
            'Formato de resposta (obrigatório), uma lista JSON neste formato exato:\n'
            '[\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}},\n'
            '  {{"tom": "*", "probabilidade": *, "ideal_quando": "texto aqui", "resposta": "texto aqui"}}\n'
            ']'
        )
    else:
        raise ValueError('As configurações devem ser baseline, soft ou aggressive')

    return prompt


# MAIN
for config in testConfigs:
    for i in range(len(messages)//19):
        print('=' * 60)
        print(f'Modelo: {args.model}')
        print(f'Configuração: {config}')
        print(requestLLMResponse(customizePrompt(messages['mensagem'][i], config), args.model))
        print('=' * 60)